import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from aiohttp import web


@dataclass
class NormalizedRequest:
    protocol: Literal["anthropic", "openai_chat", "openai_responses"]
    model: str
    stream: bool
    original_messages: list[dict[str, Any]]
    original_system: str | list | None
    openai_messages: list[dict[str, Any]]
    openai_tools: list[dict[str, Any]] | None
    request_id_prefix: str
    original_body: dict[str, Any] | None = None


def _encode_sse_event(event_type: str | None, data: dict[str, Any] | str) -> bytes:
    lines: list[str] = []
    if event_type is not None:
        lines.append(f"event: {event_type}")
    if isinstance(data, str):
        lines.append(f"data: {data}")
    else:
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return ("\n".join(lines) + "\n\n").encode()


def _iter_content_blocks(content_blocks: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return list(enumerate(content_blocks))


class ProtocolAdapter:
    """Normalize protocol-specific requests and encode protocol-specific responses."""

    _OPENAI_FINISH_REASON: dict[str, str] = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }

    def normalize_request(self, path: str, body: dict[str, Any]) -> NormalizedRequest:
        if "messages" in path:
            return self._normalize_anthropic_request(body)
        if "chat/completions" in path:
            return self._normalize_openai_request(body)
        if "responses" in path:
            return self._normalize_openai_responses_request(body)
        raise ValueError(f"unsupported intercepted path: {path}")

    async def build_response(
        self,
        request: web.Request,
        normalized: NormalizedRequest,
        msg_id: str,
        parsed_reply: Any,
        input_token_count: int,
        output_token_count: int,
    ) -> web.StreamResponse:
        if normalized.protocol == "anthropic":
            return await self._anthropic_response(
                request=request,
                msg_id=msg_id,
                model=normalized.model,
                content_blocks=parsed_reply.content_blocks,
                final_stop=parsed_reply.final_stop_reason,
                n_input=input_token_count,
                n_output=output_token_count,
                client_streaming=normalized.stream,
            )
        if normalized.protocol == "openai_responses":
            return await self._openai_responses_response(
                request=request,
                normalized=normalized,
                msg_id=msg_id,
                model=normalized.model,
                content_blocks=parsed_reply.content_blocks,
                final_stop=parsed_reply.final_stop_reason,
                n_input=input_token_count,
                n_output=output_token_count,
                client_streaming=normalized.stream,
            )
        return await self._openai_response(
            request=request,
            msg_id=msg_id,
            model=normalized.model,
            content_blocks=parsed_reply.content_blocks,
            final_stop=parsed_reply.final_stop_reason,
            n_input=input_token_count,
            n_output=output_token_count,
            client_streaming=normalized.stream,
        )

    async def _anthropic_response(
        self,
        request: web.Request,
        msg_id: str,
        model: str,
        content_blocks: list[dict[str, Any]],
        final_stop: str,
        n_input: int,
        n_output: int,
        client_streaming: bool,
    ) -> web.StreamResponse:
        if client_streaming:
            sse_chunks = self._anthropic_sse_events(
                msg_id, model, content_blocks, final_stop, n_input, n_output,
            )
            stream_resp = web.StreamResponse(status=200, headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })
            await stream_resp.prepare(request)
            for chunk in sse_chunks:
                await stream_resp.write(chunk)
            await stream_resp.write_eof()
            return stream_resp

        anthropic_resp = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": final_stop,
            "stop_sequence": None,
            "usage": {
                "input_tokens": n_input,
                "output_tokens": n_output,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return web.Response(
            status=200,
            body=json.dumps(anthropic_resp, ensure_ascii=False).encode(),
            content_type="application/json",
        )

    async def _openai_response(
        self,
        request: web.Request,
        msg_id: str,
        model: str,
        content_blocks: list[dict[str, Any]],
        final_stop: str,
        n_input: int,
        n_output: int,
        client_streaming: bool,
    ) -> web.StreamResponse:
        if client_streaming:
            sse_chunks = self._openai_sse_events(
                msg_id, model, content_blocks, final_stop, n_input, n_output,
            )
            stream_resp = web.StreamResponse(status=200, headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })
            await stream_resp.prepare(request)
            for chunk in sse_chunks:
                await stream_resp.write(chunk)
            await stream_resp.write_eof()
            return stream_resp

        finish_reason = self._to_openai_finish_reason(final_stop)
        oai_resp = {
            "id": msg_id,
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": self._content_blocks_to_openai_message(content_blocks),
                         "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": n_input,
                "completion_tokens": n_output,
                "total_tokens": n_input + n_output,
            },
        }
        return web.Response(
            status=200,
            body=json.dumps(oai_resp, ensure_ascii=False).encode(),
            content_type="application/json",
        )

    async def _openai_responses_response(
        self,
        request: web.Request,
        normalized: NormalizedRequest,
        msg_id: str,
        model: str,
        content_blocks: list[dict[str, Any]],
        final_stop: str,
        n_input: int,
        n_output: int,
        client_streaming: bool,
    ) -> web.StreamResponse:
        output_items = self._content_blocks_to_responses_output(content_blocks)
        original_body = normalized.original_body or {}
        response_obj = {
            "id": msg_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output": output_items,
            "usage": {
                "input_tokens": n_input,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": n_output,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": n_input + n_output,
            },
            "error": None,
            "incomplete_details": None,
            "instructions": original_body.get("instructions"),
            "metadata": original_body.get("metadata"),
            "tools": original_body.get("tools", []),
            "tool_choice": original_body.get("tool_choice", "auto"),
            "parallel_tool_calls": original_body.get("parallel_tool_calls", True),
            "previous_response_id": original_body.get("previous_response_id"),
            "reasoning": original_body.get("reasoning"),
            "truncation": original_body.get("truncation"),
            "stop_reason": final_stop,
        }

        if client_streaming:
            sse_chunks = self._openai_responses_sse_events(response_obj)
            stream_resp = web.StreamResponse(status=200, headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })
            await stream_resp.prepare(request)
            for chunk in sse_chunks:
                await stream_resp.write(chunk)
            await stream_resp.write_eof()
            return stream_resp

        return web.Response(
            status=200,
            body=json.dumps(response_obj, ensure_ascii=False).encode(),
            content_type="application/json",
        )

    def _normalize_anthropic_request(self, body: dict[str, Any]) -> NormalizedRequest:
        system = body.get("system")
        messages = body.get("messages", [])
        tools_raw: list[dict] = body.get("tools") or []

        return NormalizedRequest(
            protocol="anthropic",
            model=body.get("model", ""),
            stream=bool(body.get("stream", False)),
            original_messages=messages,
            original_system=system,
            openai_messages=self._anthropic_to_openai_messages(system, messages),
            openai_tools=self._anthropic_tools_to_openai(tools_raw) if tools_raw else None,
            request_id_prefix="msg",
            original_body=body,
        )

    def _normalize_openai_request(self, body: dict[str, Any]) -> NormalizedRequest:
        messages: list[dict] = body.get("messages", [])
        tools_raw: list[dict] = body.get("tools") or []
        system = messages[0].get("content") if messages and messages[0].get("role") == "system" else None

        return NormalizedRequest(
            protocol="openai_chat",
            model=body.get("model", ""),
            stream=bool(body.get("stream", False)),
            original_messages=messages,
            original_system=system,
            openai_messages=messages,
            openai_tools=tools_raw if tools_raw else None,
            request_id_prefix="chatcmpl",
            original_body=body,
        )

    def _normalize_openai_responses_request(self, body: dict[str, Any]) -> NormalizedRequest:
        instructions = body.get("instructions")
        input_data = body.get("input", [])
        original_messages = list(input_data) if isinstance(input_data, list) else []
        tools_raw: list[dict[str, Any]] = body.get("tools") or []
        return NormalizedRequest(
            protocol="openai_responses",
            model=body.get("model", ""),
            stream=bool(body.get("stream", False)),
            original_messages=original_messages,
            original_system=instructions,
            openai_messages=self._openai_responses_to_openai_messages(body),
            openai_tools=self._openai_responses_tools_to_openai(tools_raw) if tools_raw else None,
            request_id_prefix="resp",
            original_body=body,
        )

    def _anthropic_to_openai_messages(self, system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        openai: list[dict[str, Any]] = []

        if system:
            if isinstance(system, str):
                openai.append({"role": "system", "content": system})
            elif isinstance(system, list):
                parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
                openai.append({"role": "system", "content": "\n".join(parts)})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                openai.append({"role": role, "content": content})
                continue

            openai_msg: dict[str, Any] = {"role": role}
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "text" and block.get("text"):
                    content_parts.append({"type": "text", "text": block["text"]})
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    })
                elif btype == "tool_result" and role == "user":
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        text_parts = [b.get("text", "") for b in inner
                                      if isinstance(b, dict) and b.get("type") == "text"]
                        tool_content = "\n".join(text_parts)
                    else:
                        tool_content = str(inner) if inner else ""
                    openai.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or block.get("id") or "",
                        "content": tool_content,
                    })

            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            if content_parts:
                openai_msg["content"] = (
                    content_parts[0]["text"]
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text"
                    else content_parts
                )
            elif not tool_calls:
                continue

            openai.append(openai_msg)

        return openai

    def _openai_responses_to_openai_messages(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        instructions = body.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_data = body.get("input", "")
        if isinstance(input_data, str):
            if input_data:
                messages.append({"role": "user", "content": input_data})
            return messages or [{"role": "user", "content": ""}]

        if not isinstance(input_data, list):
            return messages or [{"role": "user", "content": ""}]

        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type", "")
            role = item.get("role", "user")

            if item_type == "function_call":
                call_id = item.get("call_id", item.get("id", ""))
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }],
                })
                continue

            if item_type == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })
                continue

            if item_type == "local_shell_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", item.get("id", "")),
                    "content": json.dumps(item.get("output", []), ensure_ascii=False),
                })
                continue

            if item_type == "message" or "content" in item or role in ("user", "assistant", "system", "developer"):
                content = item.get("content", item.get("text", ""))
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") in ("input_text", "output_text", "text"):
                                text_parts.append(part.get("text", ""))
                            elif "text" in part:
                                text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(p for p in text_parts if p)
                msg_role = "system" if role == "developer" else role
                if content:
                    messages.append({"role": msg_role, "content": content})

        return messages or [{"role": "user", "content": ""}]

    def _openai_responses_tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            if tool.get("type") != "function":
                continue

            function_block = tool.get("function")
            if isinstance(function_block, dict):
                normalized.append(tool)
                continue

            function_payload: dict[str, Any] = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            if "strict" in tool:
                function_payload["strict"] = tool["strict"]

            normalized.append({
                "type": "function",
                "function": function_payload,
            })

        return normalized

    def _anthropic_tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    def _anthropic_sse_events(
        self,
        msg_id: str,
        model: str,
        content_blocks: list[dict[str, Any]],
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> list[bytes]:
        events: list[bytes] = []

        def emit(event_type: str, data: dict[str, Any]) -> None:
            events.append(_encode_sse_event(event_type, data))

        emit("message_start", {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model, "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }})

        for idx, block in _iter_content_blocks(content_blocks):
            btype = block.get("type", "text")
            if btype == "text":
                emit("content_block_start", {"type": "content_block_start", "index": idx,
                      "content_block": {"type": "text", "text": ""}})
                emit("ping", {"type": "ping"})
                if block.get("text"):
                    emit("content_block_delta", {"type": "content_block_delta", "index": idx,
                          "delta": {"type": "text_delta", "text": block["text"]}})
            elif btype == "thinking":
                emit("content_block_start", {"type": "content_block_start", "index": idx,
                      "content_block": {"type": "thinking", "thinking": ""}})
                if block.get("thinking"):
                    emit("content_block_delta", {"type": "content_block_delta", "index": idx,
                          "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
            elif btype == "tool_use":
                emit("content_block_start", {"type": "content_block_start", "index": idx,
                      "content_block": {"type": "tool_use", "id": block.get("id", ""),
                                        "name": block.get("name", ""), "input": {}}})
                emit("content_block_delta", {"type": "content_block_delta", "index": idx,
                      "delta": {"type": "input_json_delta",
                                "partial_json": json.dumps(block.get("input", {}))}})
            emit("content_block_stop", {"type": "content_block_stop", "index": idx})

        emit("message_delta", {"type": "message_delta",
              "delta": {"stop_reason": stop_reason, "stop_sequence": None},
              "usage": {"output_tokens": output_tokens}})
        emit("message_stop", {"type": "message_stop"})
        return events

    def _to_openai_finish_reason(self, stop_reason: str) -> str:
        return self._OPENAI_FINISH_REASON.get(stop_reason, "stop")

    def _content_blocks_to_openai_message(self, content_blocks: list[dict[str, Any]]) -> dict[str, Any]:
        text = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        tool_calls = [
            {
                "id": b.get("id", ""),
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                },
            }
            for b in content_blocks if b.get("type") == "tool_use"
        ]
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _content_blocks_to_responses_output(self, content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        text = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text").strip()
        if text:
            output_items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            })

        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            call_id = block.get("id", f"call_{uuid.uuid4().hex[:24]}")
            output_items.append({
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                "status": "completed",
            })

        return output_items

    def _openai_sse_events(
        self,
        msg_id: str,
        model: str,
        content_blocks: list[dict[str, Any]],
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> list[bytes]:
        finish_reason = self._to_openai_finish_reason(stop_reason)
        chunk_base = {"id": msg_id, "object": "chat.completion.chunk", "model": model}
        events: list[bytes] = []

        def emit(data: dict[str, Any]) -> None:
            events.append(_encode_sse_event(None, {**chunk_base, **data}))

        emit({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

        text = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        if text:
            emit({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})

        for i, b in enumerate(b for b in content_blocks if b.get("type") == "tool_use"):
            emit({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": b.get("id", ""),
                  "type": "function", "function": {"name": b.get("name", ""), "arguments": ""}}]},
                  "finish_reason": None}]})
            emit({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function":
                  {"arguments": json.dumps(b.get("input", {}), ensure_ascii=False)}}]},
                  "finish_reason": None}]})

        emit({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
              "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens}})
        events.append(_encode_sse_event(None, "[DONE]"))
        return events

    def _openai_responses_sse_events(self, result: dict[str, Any]) -> list[bytes]:
        events: list[bytes] = []

        def emit(event_type: str, data: dict[str, Any]) -> None:
            events.append(_encode_sse_event(event_type, data))

        emit("response.created", result)
        emit("response.in_progress", {**result, "status": "in_progress"})

        for output_index, item in enumerate(result.get("output", [])):
            emit("response.output_item.added", {"item": item, "output_index": output_index})

            if item.get("type") == "message":
                for content_index, part in enumerate(item.get("content", [])):
                    emit("response.content_part.added", {
                        "part": {**part, "text": ""},
                        "output_index": output_index,
                        "content_index": content_index,
                    })
                    if part.get("type") == "output_text":
                        text = part.get("text", "")
                        emit("response.output_text.delta", {
                            "delta": text,
                            "output_index": output_index,
                            "content_index": content_index,
                        })
                        emit("response.output_text.done", {
                            "text": text,
                            "output_index": output_index,
                            "content_index": content_index,
                        })
                    emit("response.content_part.done", {
                        "part": part,
                        "output_index": output_index,
                        "content_index": content_index,
                    })

            emit("response.output_item.done", {"item": item, "output_index": output_index})

        emit("response.completed", result)
        return events
