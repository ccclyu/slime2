import dataclasses
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

try:
    from .protocol_adapter import NormalizedRequest, ProtocolAdapter
except ImportError:
    from protocol_adapter import NormalizedRequest, ProtocolAdapter

logger = logging.getLogger(__name__)

_VALID_STOP_REASONS = {"end_turn", "tool_use", "max_tokens"}


def _validate_stop_reason(stop_reason: str) -> str:
    if stop_reason not in _VALID_STOP_REASONS:
        raise ValueError(f"invalid stop_reason: {stop_reason}")
    return stop_reason


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AgentTurn:
    request_messages: list[dict[str, Any]]
    request_system: str | list | None

    input_ids: list[int]    # tokens fed to the model this turn (includes gen prompt)
    output_ids: list[int]   # tokens the model generated
    output_log_probs: list[float]  # per-output-token log prob; len == len(output_ids)

    # Decoded response returned to agent
    response_content: list[dict[str, Any]]
    stop_reason: str = "end_turn"

    def __post_init__(self) -> None:
        if len(self.output_log_probs) != len(self.output_ids):
            raise ValueError("output_log_probs must align with output_ids")
        self.stop_reason = _validate_stop_reason(self.stop_reason)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.request_messages

    @property
    def system(self) -> str | list | None:
        return self.request_system

    @property
    def log_probs(self) -> list[float]:
        return self.output_log_probs


@dataclass
class GenerationResult:
    input_ids: list[int]
    output_ids: list[int]
    log_probs: list[float]
    output_text: str
    sglang_stop_reason: str


@dataclass
class ParsedAssistantReply:
    content_blocks: list[dict[str, Any]]
    final_stop_reason: str

    def __post_init__(self) -> None:
        self.final_stop_reason = _validate_stop_reason(self.final_stop_reason)


@dataclass
class StoredResponseState:
    response_id: str
    parent_response_id: str | None
    openai_messages: list[dict[str, Any]]


# =============================================================================
# SGLang Transport
# =============================================================================


async def _call_sglang_generate(
    sglang_url: str,
    input_ids: list[int],
    sampling_params: dict,
    routing_key: str | None = None,
) -> GenerationResult:

    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    headers = {}
    if routing_key:
        headers["X-SMG-Routing-Key"] = routing_key

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{sglang_url}/generate",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"SGLang /generate returned {resp.status}: {body[:300]}"
                )
            data = await resp.json()

    output_text = data.get("text", "")
    meta = data.get("meta_info", {})

    output_ids: list[int] = [int(x) for x in (data.get("output_ids") or [])]

    token_logprobs: list = meta.get("output_token_logprobs") or []
    log_probs: list[float] = [float(item[0]) for item in token_logprobs]

    # If logprobs are shorter than output_ids (shouldn't happen, but be safe)
    if len(log_probs) < len(output_ids):
        log_probs.extend([0.0] * (len(output_ids) - len(log_probs)))

    finish_reason = meta.get("finish_reason", {})
    fr_type = finish_reason.get("type", "stop") if isinstance(finish_reason, dict) else str(finish_reason)
    stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(fr_type, "end_turn")

    return GenerationResult(
        input_ids=input_ids,
        output_ids=output_ids,
        log_probs=log_probs,
        output_text=output_text,
        sglang_stop_reason=stop_reason,
    )

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_tool_calls_local(text: str) -> tuple[str, list[dict]]:
    """Parse Qwen-style <tool_call>...</tool_call> blocks from generated text."""
    calls: list[dict] = []
    normal_parts: list[str] = []
    last_end = 0
    for m in _TOOL_CALL_RE.finditer(text):
        normal_parts.append(text[last_end:m.start()])
        last_end = m.end()
        try:
            call = json.loads(m.group(1))
            calls.append({
                "name": call.get("name", ""),
                "arguments": call.get("arguments", call.get("parameters", {})),
            })
        except json.JSONDecodeError:
            normal_parts.append(m.group(0))
    normal_parts.append(text[last_end:])
    return "".join(normal_parts).strip(), calls


async def _call_sglang_parse_function_call(
    sglang_url: str,
    text: str,
    tools: list[dict] | None,
    tool_call_parser: str,
) -> tuple[str, list[dict]]:
    """Parse tool calls from SGLang's generated text.

    Falls back to local Qwen-format parsing if the SGLang endpoint returns 404
    (endpoint requires --tool-call-parser on the server, which may not be set).
    """
    payload: dict[str, Any] = {"text": text, "tool_call_parser": tool_call_parser}
    if tools:
        payload["tools"] = tools

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{sglang_url}/parse_function_call",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 404:
                    return _parse_tool_calls_local(text)
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"SGLang /parse_function_call returned {resp.status}: {body[:300]}"
                    )
                data = await resp.json()
        return data.get("normal_text", ""), data.get("calls", [])
    except aiohttp.ClientConnectorError:
        return _parse_tool_calls_local(text)


# =============================================================================
# Token Preparation
# =============================================================================


class TurnAccumulator:
    """Maintain exact token state across turns for one proxied conversation."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._accumulated_ids: list[int] | None = None
        self._prev_openai_msg_count: int = 0

    def clear(self) -> None:
        self._accumulated_ids = None
        self._prev_openai_msg_count = 0

    def get_input_ids(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        """Update and return the exact model input IDs for the current turn."""
        if self._accumulated_ids is None:
            self._accumulated_ids = self._render_input_ids(openai_messages, openai_tools)
        else:
            full_input_ids = self._render_input_ids(openai_messages, openai_tools)
            accumulated_len = len(self._accumulated_ids)
            if full_input_ids[:accumulated_len] == self._accumulated_ids:
                self._accumulated_ids = full_input_ids
            else:
                new_messages = openai_messages[self._prev_openai_msg_count:]
                new_tool_messages = [m for m in new_messages if m.get("role") != "assistant"]
                if new_tool_messages:
                    tool_delta = self.compute_new_message_tokens(new_tool_messages)
                    self._accumulated_ids = self._accumulated_ids + tool_delta
        self._prev_openai_msg_count = len(openai_messages)
        return list(self._accumulated_ids)

    def _render_input_ids(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        tok_kwargs: dict[str, Any] = dict(
            tokenize=True,
            add_special_tokens=True,
            add_generation_prompt=True,
        )
        if openai_tools:
            tok_kwargs["tools"] = openai_tools
        return list(
            self.tokenizer.apply_chat_template(
                self._flatten_messages(openai_messages), **tok_kwargs
            )
        )

    def append_output_ids(self, output_ids: list[int]) -> None:
        """Extend accumulated state with raw model output tokens."""
        if self._accumulated_ids is None:
            raise RuntimeError("append_output_ids called before get_input_ids")
        self._accumulated_ids = self._accumulated_ids + list(output_ids)

    def compute_new_message_tokens(self, new_messages: list[dict]) -> list[int]:
        """Return the continuation delta for newly appended non-assistant messages."""
        stub: list[dict] = [
            {"role": "user", "content": "\x00"},
            {"role": "assistant", "content": "\x00"},
        ]

        stub_ids: list[int] = list(self.tokenizer.apply_chat_template(
            stub,
            tokenize=True,
            add_special_tokens=True,
            add_generation_prompt=False,
        ))
        extended_ids: list[int] = self.tokenizer.apply_chat_template(
            self._flatten_messages(stub + new_messages),
            tokenize=True,
            add_special_tokens=True,
            add_generation_prompt=True,
        )

        im_end_id: int = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        last_im_end = next(i for i in reversed(range(len(stub_ids))) if stub_ids[i] == im_end_id)
        return extended_ids[last_im_end + 1:]

    def _flatten_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        parts.append(str(block))
                    elif block.get("type") != "image_url":
                        parts.append(block.get("text", ""))
                flat = {**msg, "content": "\n".join(p for p in parts if p)}
            else:
                flat = msg
            flattened.append(flat)
        return flattened


class SGLangTurnEngine:
    """Run one turn against SGLang and parse the resulting assistant reply."""

    _THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

    def __init__(
        self,
        sglang_url: str,
        sampling_params: dict[str, Any],
        tool_call_parser: str,
    ) -> None:
        self.sglang_url = sglang_url
        self.sampling_params = sampling_params
        self.tool_call_parser = tool_call_parser

    async def run_turn(
        self,
        input_ids: list[int],
        tools: list[dict[str, Any]] | None,
        routing_key: str | None,
    ) -> tuple[GenerationResult, ParsedAssistantReply]:
        generation = await _call_sglang_generate(
            self.sglang_url,
            input_ids,
            self.sampling_params,
            routing_key=routing_key,
        )
        if tools:
            normal_text, calls = await _call_sglang_parse_function_call(
                self.sglang_url,
                generation.output_text,
                tools,
                self.tool_call_parser,
            )
        else:
            normal_text, calls = generation.output_text, []
        parsed_reply = self._build_content_blocks(normal_text, calls)
        if generation.sglang_stop_reason == "max_tokens":
            parsed_reply.final_stop_reason = "max_tokens"
        return generation, parsed_reply

    def _build_content_blocks(
        self,
        normal_text: str,
        calls: list[dict[str, Any]],
    ) -> ParsedAssistantReply:
        content_blocks: list[dict[str, Any]] = []

        remaining = normal_text
        for match in self._THINK_RE.finditer(normal_text):
            content_blocks.append({"type": "thinking", "thinking": match.group(1)})
            remaining = remaining.replace(match.group(0), "", 1)

        if remaining.strip():
            content_blocks.append({"type": "text", "text": remaining.strip()})

        for call in calls:
            args: Any = call.get("parameters", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            content_blocks.append({
                "type": "tool_use",
                "id": f"toolu_{(call.get('tool_index', 0) + 1):04d}",
                "name": call.get("name", ""),
                "input": args,
            })

        return ParsedAssistantReply(
            content_blocks=content_blocks,
            final_stop_reason="tool_use" if calls else "end_turn",
        )


class ModelInterceptProxy:
    """Proxy agent requests through SGLang while capturing exact token-level turns."""

    def __init__(
        self,
        port: int,
        sglang_url: str,
        tokenizer: Any,
        sampling_params: dict,
        tool_call_parser: str = "qwen",
    ) -> None:
        self.port = port
        self.sglang_url = sglang_url.rstrip("/")
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.tool_call_parser = tool_call_parser
        self._turns: list[AgentTurn] = []
        self._runner: web.AppRunner | None = None
        self._accumulator = TurnAccumulator(tokenizer)
        self._responses_store: dict[str, StoredResponseState] = {}
        self._protocol_adapter = ProtocolAdapter()
        self._turn_engine = SGLangTurnEngine(
            sglang_url=self.sglang_url,
            sampling_params=self.sampling_params,
            tool_call_parser=self.tool_call_parser,
        )
        self._routing_key: str = str(uuid.uuid4())

    @property
    def turns(self) -> list[AgentTurn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._accumulator.clear()
        self._responses_store.clear()

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(
            "ModelInterceptProxy :%d → %s (token-in/token-out via /generate)",
            self.port,
            self.sglang_url,
        )

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        path = request.match_info.get("path_info", "")
        body_bytes = await request.read()
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, ValueError):
            body = {}

        if request.method == "POST":
            if "messages" in path and "messages" in body:
                return await self._handle_intercepted_request(request, path, body)
            if "chat/completions" in path and "messages" in body:
                return await self._handle_intercepted_request(request, path, body)
            if "responses" in path:
                return await self._handle_intercepted_request(request, path, body)
        return await self._forward(request, path, body_bytes)

    async def _handle_intercepted_request(
        self,
        request: web.Request,
        path: str,
        body: dict[str, Any],
    ) -> web.StreamResponse:
        normalized = self._protocol_adapter.normalize_request(path, body)
        if normalized.protocol == "openai_responses":
            normalized = self._resolve_responses_conversation(normalized)
        return await self._handle_normalized_request(request, normalized)

    def _resolve_responses_conversation(self, normalized: NormalizedRequest) -> NormalizedRequest:
        """Resolve a Responses request into the effective full conversation.

        Minimal implementation:
        - first turn: use request-local normalized messages
        - continuation: load stored conversation from previous_response_id and append delta input
        """
        body = normalized.original_body or {}
        prev_id = body.get("previous_response_id")
        logger.info(
            "[responses_conv] incoming openai_messages=%d roles=%s prev_id=%s",
            len(normalized.openai_messages),
            [m.get("role") for m in normalized.openai_messages],
            prev_id,
        )
        if not prev_id:
            return normalized

        state = self._responses_store.get(prev_id)
        if state is None:
            raise web.HTTPBadRequest(text=f"unknown previous_response_id: {prev_id}")

        delta = list(normalized.openai_messages)
        if delta and delta[0].get("role") == "system":
            delta = delta[1:]

        merged = list(state.openai_messages) + delta
        logger.info("[responses_conv] resolved_messages=%d", len(merged))
        return dataclasses.replace(normalized, openai_messages=merged)

    async def _handle_normalized_request(
        self,
        request: web.Request,
        normalized: NormalizedRequest,
    ) -> web.StreamResponse:
        try:
            generation, parsed_reply, msg_id = await self._process_turn(normalized)
        except Exception as exc:
            logger.error("SGLang /generate failed: %s", exc, exc_info=True)
            return web.Response(status=502, text=f"SGLang error: {exc}")

        if normalized.protocol == "openai_responses":
            prev_id = (normalized.original_body or {}).get("previous_response_id")
            asst_msg = self._protocol_adapter._content_blocks_to_openai_message(
                parsed_reply.content_blocks
            )
            full_conv = list(normalized.openai_messages) + [asst_msg]
            self._responses_store[msg_id] = StoredResponseState(
                response_id=msg_id,
                parent_response_id=prev_id,
                openai_messages=full_conv,
            )

        return await self._protocol_adapter.build_response(
            request,
            normalized,
            msg_id,
            parsed_reply,
            len(generation.input_ids),
            len(generation.output_ids),
        )

    # ---- Shared: generate + capture ----

    async def _process_turn(
        self,
        normalized: NormalizedRequest,
    ) -> tuple[GenerationResult, ParsedAssistantReply, str]:
        """Run one normalized turn through tokenization, generation, parsing, and capture."""
        input_ids = self._accumulator.get_input_ids(
            normalized.openai_messages,
            normalized.openai_tools,
        )
        generation, parsed_reply = await self._turn_engine.run_turn(
            input_ids,
            normalized.openai_tools,
            routing_key=self._routing_key,
        )

        self._accumulator.append_output_ids(generation.output_ids)

        msg_id = f"{normalized.request_id_prefix}_{uuid.uuid4().hex[:24]}"
        self._turns.append(AgentTurn(
            request_messages=normalized.original_messages,
            request_system=normalized.original_system,
            input_ids=input_ids,
            output_ids=generation.output_ids,
            output_log_probs=generation.log_probs,
            response_content=parsed_reply.content_blocks,
            stop_reason=parsed_reply.final_stop_reason,
        ))
        return generation, parsed_reply, msg_id

    def _compute_new_message_tokens(
        self,
        new_messages: list[dict],
        tools: list[dict] | None,
    ) -> list[int]:
        del tools
        return self._accumulator.compute_new_message_tokens(new_messages)

    async def _forward(
        self, request: web.Request, path: str, body_bytes: bytes
    ) -> web.Response:
        url = f"{self.sglang_url}/{path}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=url,
                data=body_bytes,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as upstream:
                resp_bytes = await upstream.read()
                resp_headers = {
                    k: v for k, v in upstream.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "content-type")
                }
                return web.Response(
                    status=upstream.status,
                    body=resp_bytes,
                    content_type=upstream.content_type,
                    headers=resp_headers,
                )
