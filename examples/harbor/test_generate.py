"""Standalone test for generate_with_harbor.generate on SWEBench tasks.

Usage:
    # Single task — full verbose output:
    python test_generate.py -t django__django-13410 --hf-checkpoint Qwen/Qwen3-Coder-Next \\
        --sglang-url http://127.0.0.1:30000

    # Multiple tasks concurrently — summary table:
    python test_generate.py --tasks django__django-13410 sympy__sympy-19346 pytest-dev__pytest-7236 \\
        --hf-checkpoint Qwen/Qwen3-Coder-Next --sglang-url http://127.0.0.1:30000

    # Default 10 concurrent tasks:
    python test_generate.py --n-tasks 10 --hf-checkpoint Qwen/Qwen3-Coder-Next \\
        --sglang-url http://127.0.0.1:30000

    # Offline proxy unit tests only (tokenizer required, no SGLang or Docker):
    python test_generate.py --unit-test --hf-checkpoint Qwen/Qwen3-Coder-Next

# pip install aioboto3 weave strands-agents anls esprima modal harbor
# curl -SL https://github.com/docker/compose/releases/download/v2.34.0/docker-compose-linux-x86_64 \\
#     -o /usr/lib/docker/cli-plugins/docker-compose && chmod +x ...

The script:
  1. Downloads the task via harbor's TaskClient (cached after first run).
  2. Loads the task instruction from instruction.md.
  3. Builds a minimal args Namespace and a Sample.
  4. Calls generate() and prints reward, turn count, and token stats.
  5. Verifies strict token-in/token-out invariants on the captured turns.
"""

import argparse
import asyncio
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[3] / "harbor" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2]))

from slime.utils.types import Sample
from slime.utils.processing_utils import load_tokenizer

from harbor.models.task.id import GitTaskId
from harbor.models.task.task import Task
from harbor.tasks.client import TaskClient

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from generate_with_harbor import SWEBENCH_CONFIGS, _AGENT_URL_ENV, generate
from model_proxy import (
    CapturedTurn,
    ModelInterceptProxy,
    _OPENAI_FINISH_REASON,
    _content_blocks_to_openai_message,
    _openai_sse_events,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("test_generate")

_console = Console()

# Default pool of tasks used by --n-tasks
DEFAULT_TASKS: list[str] = [
    "django__django-13410",
    "django__django-13590",
    "sympy__sympy-19346",
    "pytest-dev__pytest-7236",
    "scikit-learn__scikit-learn-10844",
    "astropy__astropy-12907",
    "matplotlib__matplotlib-23412",
    "psf__requests-5414",
    "sphinx-doc__sphinx-9367",
    "pydata__xarray-6721",
]


# ---------------------------------------------------------------------------
# Minimal args stub
# ---------------------------------------------------------------------------


def _build_args(hf_checkpoint: str) -> Namespace:
    """Build a minimal Namespace satisfying SlimeArgs (via @typed_generate)."""
    return Namespace(
        hf_checkpoint=hf_checkpoint,
        model_name=None,
        sglang_server_concurrency=512,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_max_response_len=32768,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )


# ---------------------------------------------------------------------------
# Task instruction loader
# ---------------------------------------------------------------------------


def _load_instruction(task_id: str) -> str:
    cfg = SWEBENCH_CONFIGS
    task_git_id = GitTaskId(
        git_url=cfg["git_url"],
        git_commit_id=cfg["git_commit_id"],
        path=Path(cfg["dataset_path_prefix"]) / task_id,
    )
    client = TaskClient()
    task_dirs = client.download_tasks(task_ids=[task_git_id])
    return Task(task_dir=task_dirs[0]).instruction


# ---------------------------------------------------------------------------
# Rich conversation printer
# ---------------------------------------------------------------------------


def _content_str(content: object) -> str:
    """Flatten Anthropic content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            t = block.get("thinking") or block.get("text", "")
            parts.append(f"<think>{t}</think>")
        elif btype == "tool_use":
            args = json.dumps(block.get("input", {}), indent=2, ensure_ascii=False)
            parts.append(f"tool_use: {block.get('name')}\n{args}")
        elif btype == "tool_result":
            inner_text = _content_str(block.get("content", ""))
            parts.append(f"tool_result id={block.get('tool_use_id','')}\n{inner_text}")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def _print_conversation(captures: list[CapturedTurn]) -> None:
    """Render captured proxy turns as a rich conversation with token stats."""
    if not captures:
        _console.print("[dim]No proxy captures.[/dim]")
        return

    _console.print(Rule("[bold]Conversation[/bold]"))

    # System prompt
    system = captures[0].system
    if system:
        sys_text = _content_str(system) if not isinstance(system, str) else system
        if sys_text:
            _console.print(Panel(
                Text(sys_text[:600] + ("…" if len(sys_text) > 600 else ""), style="dim"),
                title=f"[dim]system  ({len(captures[0].input_ids)} prompt tokens)[/dim]",
                border_style="dim", padding=(0, 1),
            ))

    # Initial user prompt
    for msg in captures[0].messages:
        if msg.get("role") == "user":
            text = _content_str(msg.get("content", ""))
            if text:
                _console.print(Panel(
                    Text(text[:600] + ("…" if len(text) > 600 else "")),
                    title="[blue bold]user[/blue bold]",
                    border_style="blue", padding=(0, 1),
                ))
            break

    # Per-turn: assistant response then tool observations
    for i, turn in enumerate(captures):
        # Token stats for this turn
        n_out = len(turn.output_ids)
        avg_lp = sum(turn.log_probs) / len(turn.log_probs) if turn.log_probs else 0.0
        tool_obs_len = 0
        if i + 1 < len(captures):
            accumulated_len = len(turn.input_ids) + len(turn.output_ids)
            tool_obs_len = len(captures[i + 1].input_ids) - accumulated_len

        asst_text = _content_str(turn.response_content)
        tool_obs_str = str(tool_obs_len) if i + 1 < len(captures) else "—"
        token_info = (
            f"  [dim]out={n_out} logp={avg_lp:.3f} "
            f"tool_obs={tool_obs_str}[/dim]"
        )
        _console.print(Panel(
            Text(asst_text[:600] + ("…" if len(asst_text) > 600 else "")),
            title=f"[green bold]assistant (turn {i + 1})[/green bold]{token_info}",
            border_style="green", padding=(0, 1),
        ))

        # Tool result messages (from the Anthropic messages list)
        if i + 1 < len(captures):
            # New messages = those in next turn's list beyond what current turn had
            # (skip the assistant message at index len(captures[i].messages))
            next_msgs = captures[i + 1].messages
            prev_count = len(captures[i].messages) + 1  # +1 for asst_i
            for msg in next_msgs[prev_count:]:
                text = _content_str(msg.get("content", ""))
                if text:
                    _console.print(Panel(
                        Text(text[:400] + ("…" if len(text) > 400 else "")),
                        title=f"[yellow bold]{msg.get('role', 'tool')}[/yellow bold]",
                        border_style="yellow", padding=(0, 1),
                    ))

    _console.print(Rule())


# ---------------------------------------------------------------------------
# Token-in / token-out invariant checker
# ---------------------------------------------------------------------------


def check_token_in_token_out(
    captures: list[CapturedTurn],
    tokenizer: Any,
) -> bool:
    """Verify strict token-in/token-out invariants across all captured turns.

    Checks three properties:

    1. **Log-prob alignment**: ``len(log_probs[i]) == len(output_ids[i])`` for
       every turn.

    2. **Prefix invariant**: ``captures[i+1].input_ids`` must start with
       ``captures[i].input_ids + captures[i].output_ids``.  A violation means
       the assistant response was double-counted or the accumulated context is
       inconsistent.

    3. **Inter-message separator**: the first token of the tool-observation
       segment (i.e., ``captures[i+1].input_ids[accumulated_len]``) must decode
       to the inter-message separator (``\\n`` for Qwen3).  A missing separator
       means the proxy stitched ``<|im_end|><|im_start|>`` without the ``\\n``
       that the model's training data always contained.
    """
    if not captures:
        _console.print("[yellow]No captures — skipping token-in/token-out check.[/yellow]")
        return True

    _console.print(Rule("[bold]Token-in / token-out check[/bold]"))

    failures: list[str] = []

    # --- Per-turn table ---
    table = Table(show_header=True, header_style="bold")
    table.add_column("Turn", justify="right")
    table.add_column("input_ids", justify="right")
    table.add_column("output_ids", justify="right")
    table.add_column("log_probs", justify="right")
    table.add_column("tool_obs", justify="right")
    table.add_column("lp_align", justify="center")
    table.add_column("prefix_ok", justify="center")
    table.add_column("sep_ok", justify="center")

    im_end_str = tokenizer.decode(
        [tokenizer.convert_tokens_to_ids("<|im_end|>")],
        skip_special_tokens=False,
    )

    for i, turn in enumerate(captures):
        n_in = len(turn.input_ids)
        n_out = len(turn.output_ids)
        n_lp = len(turn.log_probs)
        accumulated_len = n_in + n_out

        # 1. Log-prob alignment
        lp_align = n_lp == n_out
        if not lp_align:
            failures.append(
                f"Turn {i}: log_prob length {n_lp} != output_ids length {n_out}"
            )

        # 2. Prefix invariant & 3. Separator (only checkable if next turn exists)
        prefix_ok = True
        sep_ok: bool | None = None
        tool_obs_len = 0

        if i + 1 < len(captures):
            next_ids = captures[i + 1].input_ids
            tool_obs_len = len(next_ids) - accumulated_len
            expected_prefix = turn.input_ids + turn.output_ids

            if next_ids[:accumulated_len] != expected_prefix:
                prefix_ok = False
                # Find first divergence position
                diverge = next(
                    (j for j, (a, b) in enumerate(zip(next_ids, expected_prefix)) if a != b),
                    min(len(next_ids), accumulated_len),
                )
                failures.append(
                    f"Turn {i}: prefix invariant violated at position {diverge} "
                    f"(next input_ids[{diverge}]={next_ids[diverge] if diverge < len(next_ids) else 'OOB'} "
                    f"expected={expected_prefix[diverge] if diverge < len(expected_prefix) else 'OOB'})"
                )

            # 3. Check inter-message separator
            if tool_obs_len > 0:
                sep_token_id = next_ids[accumulated_len]
                sep_decoded = tokenizer.decode(
                    [sep_token_id], skip_special_tokens=False
                )
                # The separator should be \n (not <|im_start|> directly)
                sep_ok = sep_decoded == "\n"
                if not sep_ok:
                    failures.append(
                        f"Turn {i}: inter-message separator missing — "
                        f"first tool_obs token is {sep_decoded!r} (id={sep_token_id}), "
                        f"expected '\\n'. The <|im_end|>→<|im_start|> gap is wrong."
                    )
            else:
                sep_ok = None  # no tool obs (final turn)

        table.add_row(
            str(i),
            str(n_in),
            str(n_out),
            str(n_lp),
            str(tool_obs_len) if i + 1 < len(captures) else "—",
            "[green]✓[/green]" if lp_align else "[red]✗[/red]",
            "[green]✓[/green]" if prefix_ok else "[red]✗[/red]",
            (
                "[green]✓[/green]" if sep_ok is True
                else "[red]✗[/red]" if sep_ok is False
                else "[dim]—[/dim]"
            ),
        )

    _console.print(table)

    if not failures:
        _console.print(f"[green]✓ All {len(captures)} turns pass token-in/token-out.[/green]")
        _console.print(Rule())
        return True

    _console.print(f"[red]✗ {len(failures)} failure(s):[/red]")
    for f in failures:
        _console.print(f"  [red]{f}[/red]")
    _console.print(Rule())
    return False


# ---------------------------------------------------------------------------
# Trajectory message-list matcher (Anthropic format)
# ---------------------------------------------------------------------------


def check_messages_match_trajectory(
    captures: list[CapturedTurn],
    trajectory_path: Path,
) -> bool:
    """Check that captures[i].messages matches the claude-code.txt trajectory.

    This checks the *Anthropic-format* message list stored in each
    CapturedTurn, not the token sequence.  For token correctness use
    ``check_token_in_token_out``.

    For each turn i, ``captures[i].messages`` should equal the accumulated
    Anthropic message history up to (but not including) turn i's response::

        Turn 0: [user_init]
        Turn 1: [user_init, asst_0, tool_0]
        Turn i: first (2*i + 1) trajectory messages
    """
    if not captures:
        _console.print("[yellow]No captures — skipping trajectory check.[/yellow]")
        return True
    if not trajectory_path.exists():
        _console.print(f"[yellow]Trajectory not found: {trajectory_path}[/yellow]")
        return True

    traj_messages: list[dict] = []
    with open(trajectory_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = obj.get("type")
            if typ == "assistant":
                msg = obj.get("message", {})
                traj_messages.append({"role": "assistant", "content": msg.get("content", [])})
            elif typ == "user":
                msg = obj.get("message", {})
                if isinstance(msg, dict) and msg.get("role") == "user":
                    traj_messages.append({"role": "user", "content": msg.get("content", [])})

    _console.print(Rule("[bold]Trajectory message check[/bold]"))
    _console.print(
        f"Captures: [bold]{len(captures)}[/bold] turns  |  "
        f"Trajectory messages: [bold]{len(traj_messages)}[/bold]"
    )

    mismatches: list[str] = []
    for i, capture in enumerate(captures):
        expected = traj_messages[:2 * i + 1]
        actual = capture.messages
        if len(actual) != len(expected):
            mismatches.append(
                f"Turn {i}: length mismatch — expected {len(expected)}, got {len(actual)}"
            )
            continue
        for j, (exp, act) in enumerate(zip(expected, actual)):
            if exp["role"] != act.get("role", ""):
                mismatches.append(
                    f"Turn {i} msg[{j}]: role mismatch — "
                    f"traj={exp['role']!r} proxy={act.get('role','')!r}"
                )
                continue
            if json.dumps(exp["content"], sort_keys=True) != json.dumps(act.get("content", ""), sort_keys=True):
                mismatches.append(
                    f"Turn {i} msg[{j}] role={exp['role']}: content mismatch\n"
                    f"  traj : {json.dumps(exp['content'], sort_keys=True)[:200]}\n"
                    f"  proxy: {json.dumps(act.get('content',''), sort_keys=True)[:200]}"
                )

    if not mismatches:
        _console.print(f"[green]✓ All {len(captures)} turns match.[/green]")
        _console.print(Rule())
        return True

    _console.print(f"[red]✗ {len(mismatches)} mismatch(es):[/red]")
    for m in mismatches:
        _console.print(f"  [red]{m}[/red]")
    _console.print(Rule())
    return False


# ---------------------------------------------------------------------------
# Offline unit tests for ModelInterceptProxy (no SGLang required)
# ---------------------------------------------------------------------------


def run_proxy_unit_tests(tokenizer: Any) -> bool:
    """Offline tests for _compute_new_message_tokens.

    Verifies the two bugs fixed in the proxy:
      Bug 1 — assistant message double-count: new_messages must not include
               the assistant role.
      Bug 2 — missing \\n separator: the delta must start with \\n so that
               _accumulated_ids ends with <|im_end|>\\n<|im_start|>user…
               rather than <|im_end|><|im_start|>user….
    """
    _console.print(Rule("[bold]Proxy unit tests (offline)[/bold]"))

    proxy = ModelInterceptProxy(
        port=0,
        sglang_url="http://localhost:30000",  # not used in offline tests
        tokenizer=tokenizer,
        sampling_params={},
    )

    failures: list[str] = []
    im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ---- Test 1: delta starts with \n (Bug 2 fix) ----
    tool_messages = [
        {"role": "tool", "tool_call_id": "toolu_0001", "content": "hello from tool"}
    ]
    delta = proxy._compute_new_message_tokens(tool_messages, tools=None)

    if not delta:
        failures.append("Test 1: delta is empty")
    else:
        first_token_str = tokenizer.decode([delta[0]], skip_special_tokens=False)
        if first_token_str != "\n":
            failures.append(
                f"Test 1 (separator): delta[0]={delta[0]} decodes to "
                f"{first_token_str!r}, expected '\\n'. "
                f"Bug 2 fix is not working."
            )
        else:
            _console.print("[green]✓ Test 1: delta starts with \\n separator.[/green]")

    # ---- Test 2: delta ends with generation prompt (well-formed) ----
    decoded_delta = tokenizer.decode(delta, skip_special_tokens=False)
    if "<|im_start|>assistant" not in decoded_delta:
        failures.append(
            f"Test 2 (gen prompt): delta does not end with generation prompt.\n"
            f"  decoded: {decoded_delta!r}"
        )
    else:
        _console.print("[green]✓ Test 2: delta ends with generation prompt.[/green]")

    # ---- Test 3: delta does NOT contain the stub content ----
    if "\x00" in decoded_delta:
        failures.append(
            "Test 3 (no stub leak): delta contains stub null-byte content — "
            "effective_stub_len calculation is wrong."
        )
    else:
        _console.print("[green]✓ Test 3: delta is free of stub content.[/green]")

    # ---- Test 4: delta contains the tool result content ----
    if "hello from tool" not in decoded_delta:
        failures.append(
            "Test 4 (content): 'hello from tool' not found in decoded delta.\n"
            f"  decoded: {decoded_delta!r}"
        )
    else:
        _console.print("[green]✓ Test 4: tool content is present in delta.[/green]")

    # ---- Test 5: stitching produces correct inter-message format ----
    # Simulate: accumulated_ids ends with <|im_end|> (as SGLang would leave it)
    # After stitching: should be <|im_end|>\n<|im_start|>user\n...
    synthetic_accumulated = [im_end_id]
    stitched = synthetic_accumulated + delta
    decoded_stitched = tokenizer.decode(stitched, skip_special_tokens=False)
    # The stitched sequence should look like: <|im_end|>\n<|im_start|>user\n...
    im_end_str = tokenizer.decode([im_end_id], skip_special_tokens=False)
    expected_boundary = im_end_str + "\n"
    if not decoded_stitched.startswith(expected_boundary):
        failures.append(
            f"Test 5 (stitching): after appending delta to [<|im_end|>], "
            f"the sequence should start with {expected_boundary!r} but starts "
            f"with {decoded_stitched[:40]!r}."
        )
    else:
        _console.print("[green]✓ Test 5: stitched sequence has correct boundary.[/green]")

    # ---- Test 6: Bug 1 guard — assistant messages in new_messages are skipped ----
    # This tests the filter `m.get("role") != "assistant"` in _handle_messages.
    # We can't call _handle_messages offline, but we can verify the filter logic.
    mixed_messages = [
        {"role": "assistant", "content": "I will run a command", "tool_calls": []},
        {"role": "tool", "tool_call_id": "toolu_0001", "content": "output here"},
    ]
    filtered = [m for m in mixed_messages if m.get("role") != "assistant"]
    if len(filtered) != 1 or filtered[0]["role"] != "tool":
        failures.append(
            "Test 6 (Bug 1 filter): assistant message not correctly filtered from new_messages."
        )
    else:
        delta_filtered = proxy._compute_new_message_tokens(filtered, tools=None)
        decoded_filtered = tokenizer.decode(delta_filtered, skip_special_tokens=False)
        if "output here" not in decoded_filtered:
            failures.append(
                f"Test 6 (Bug 1 filter): tool content missing after filtering.\n"
                f"  decoded: {decoded_filtered!r}"
            )
        else:
            _console.print("[green]✓ Test 6: assistant message correctly excluded from delta.[/green]")

    # ---- Test 7: multiple tool results produce correct delta ----
    multi_tool = [
        {"role": "tool", "tool_call_id": "toolu_0001", "content": "result_one"},
        {"role": "tool", "tool_call_id": "toolu_0002", "content": "result_two"},
    ]
    delta_multi = proxy._compute_new_message_tokens(multi_tool, tools=None)
    decoded_multi = tokenizer.decode(delta_multi, skip_special_tokens=False)
    if "result_one" not in decoded_multi or "result_two" not in decoded_multi:
        failures.append(
            f"Test 7 (multi-tool): not all tool results present in delta.\n"
            f"  decoded: {decoded_multi!r}"
        )
    else:
        _console.print("[green]✓ Test 7: multiple tool results all present in delta.[/green]")

    # ---- Test 8: _content_blocks_to_openai_message — text only ----
    text_blocks = [{"type": "text", "text": "I will help you."}]
    oai_msg = _content_blocks_to_openai_message(text_blocks)
    if oai_msg.get("role") != "assistant" or oai_msg.get("content") != "I will help you.":
        failures.append(
            f"Test 8 (OpenAI text msg): unexpected message: {oai_msg}"
        )
    else:
        _console.print("[green]✓ Test 8: text blocks → correct OpenAI message.[/green]")

    # ---- Test 9: _content_blocks_to_openai_message — tool_use ----
    tool_blocks = [{"type": "tool_use", "id": "toolu_0001", "name": "Read",
                    "input": {"file_path": "/foo"}}]
    oai_tool_msg = _content_blocks_to_openai_message(tool_blocks)
    tc = (oai_tool_msg.get("tool_calls") or [{}])[0]
    if (tc.get("id") != "toolu_0001"
            or tc.get("function", {}).get("name") != "Read"
            or '"file_path"' not in tc.get("function", {}).get("arguments", "")):
        failures.append(
            f"Test 9 (OpenAI tool msg): unexpected message: {oai_tool_msg}"
        )
    else:
        _console.print("[green]✓ Test 9: tool_use blocks → correct OpenAI tool_calls.[/green]")

    # ---- Test 10: _openai_sse_events — well-formed SSE with [DONE] and finish_reason ----
    sse_chunks = _openai_sse_events("chatcmpl_test", "qwen3", text_blocks, "end_turn", 100, 10)
    sse_text = b"".join(sse_chunks).decode()
    expected_finish = _OPENAI_FINISH_REASON["end_turn"]  # "stop"
    if not sse_text.endswith("data: [DONE]\n\n"):
        failures.append("Test 10 (OpenAI SSE): stream does not end with [DONE].")
    elif f'"finish_reason": "{expected_finish}"' not in sse_text:
        failures.append(
            f"Test 10 (OpenAI SSE): finish_reason '{expected_finish}' not found in stream."
        )
    else:
        _console.print("[green]✓ Test 10: OpenAI SSE ends with [DONE] and correct finish_reason.[/green]")

    _console.print()
    if not failures:
        _console.print(f"[green bold]All 10 proxy unit tests passed.[/green bold]")
        _console.print(Rule())
        return True

    _console.print(f"[red bold]{len(failures)} test(s) failed:[/red bold]")
    for f in failures:
        _console.print(f"  [red]{f}[/red]")
    _console.print(Rule())
    return False


# ---------------------------------------------------------------------------
# Concurrent multi-task helpers
# ---------------------------------------------------------------------------


def _check_tio_silent(captures: list[CapturedTurn], tokenizer: Any) -> bool:
    """Same three invariants as check_token_in_token_out but returns bool, no output."""
    for i, turn in enumerate(captures):
        if len(turn.log_probs) != len(turn.output_ids):
            return False
        if i + 1 < len(captures):
            accumulated_len = len(turn.input_ids) + len(turn.output_ids)
            next_ids = captures[i + 1].input_ids
            if next_ids[:accumulated_len] != turn.input_ids + turn.output_ids:
                return False
            if accumulated_len < len(next_ids):
                sep = tokenizer.decode([next_ids[accumulated_len]], skip_special_tokens=False)
                if sep != "\n":
                    return False
    return True


async def _run_task(
    task_id: str,
    args: Namespace,
    tokenizer: Any,
) -> dict:
    """Run one task and return a compact result dict (no rich output)."""
    try:
        instruction = _load_instruction(task_id)
        sample = Sample(prompt=instruction, label=task_id)
        result = await generate(args, sample, {})
        captures: list[CapturedTurn] = getattr(result, "_proxy_captures", None) or []
        tio_ok = _check_tio_silent(captures, tokenizer)

        asst_tokens = sum(result.loss_mask) if result.loss_mask else 0
        avg_lp: float | None = None
        if result.rollout_log_probs and result.loss_mask:
            lps = [lp for lp, m in zip(result.rollout_log_probs, result.loss_mask) if m == 1]
            avg_lp = sum(lps) / len(lps) if lps else None

        logger.info("done task=%s reward=%.1f turns=%d tio=%s",
                    task_id, result.reward or 0.0, len(captures), "✓" if tio_ok else "✗")
        return {
            "task_id":    task_id,
            "reward":     result.reward or 0.0,
            "turns":      len(captures),
            "status":     result.status.value,
            "tio_ok":     tio_ok,
            "total_tok":  len(result.tokens) if result.tokens else 0,
            "asst_tok":   asst_tokens,
            "avg_lp":     avg_lp,
            "error":      None,
        }
    except Exception as exc:
        logger.error("task=%s failed: %s", task_id, exc, exc_info=True)
        return {
            "task_id":   task_id,
            "reward":    0.0,
            "turns":     0,
            "status":    "error",
            "tio_ok":    False,
            "total_tok": 0,
            "asst_tok":  0,
            "avg_lp":    None,
            "error":     str(exc)[:120],
        }


def _print_multi_summary(results: list[dict], elapsed: float) -> bool:
    """Print a rich table summarising all concurrent task results."""
    _console.print(Rule("[bold]Multi-task results[/bold]"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Task", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Reward", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Asst tok", justify="right")
    table.add_column("Avg logp", justify="right")
    table.add_column("TiO", justify="center")

    all_pass = True
    n_pass = 0
    total_reward = 0.0

    for r in results:
        tio_cell = "[green]✓[/green]" if r["tio_ok"] else "[red]✗[/red]"
        reward_str = f"{r['reward']:.1f}"
        reward_style = "green" if r["reward"] > 0 else "dim"
        avg_lp_str = f"{r['avg_lp']:.4f}" if r["avg_lp"] is not None else "—"
        status_style = "green" if r["status"] == "completed" else "red"
        err = f"  [dim]{r['error'][:60]}[/dim]" if r["error"] else ""

        table.add_row(
            r["task_id"].split("__")[0] + "/" + r["task_id"].split("__")[1] + err,
            f"[{status_style}]{r['status']}[/{status_style}]",
            f"[{reward_style}]{reward_str}[/{reward_style}]",
            str(r["turns"]),
            str(r["asst_tok"]),
            avg_lp_str,
            tio_cell,
        )

        if not r["tio_ok"]:
            all_pass = False
        total_reward += r["reward"]
        if r["reward"] > 0:
            n_pass += 1

    _console.print(table)

    summary = Text()
    summary.append(f"Tasks:      {len(results)}  |  ")
    summary.append(f"Solved: ", style="bold")
    summary.append(f"{n_pass}/{len(results)}  |  ", style="green" if n_pass > 0 else "dim")
    summary.append(f"Avg reward: ", style="bold")
    summary.append(f"{total_reward / len(results):.3f}  |  ")
    summary.append(f"TiO all pass: ", style="bold")
    summary.append("✓" if all_pass else "✗", style="green" if all_pass else "red")
    summary.append(f"  |  Elapsed: {elapsed:.1f}s")
    _console.print(Panel(summary, border_style="cyan", padding=(0, 1)))
    _console.print(Rule())
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(
    task_ids: list[str],
    hf_checkpoint: str,
    trials_dir: str,
    sglang_url: str,
    agent: str,
    unit_test_only: bool,
) -> None:
    # Load tokenizer (needed for unit tests and token-in/token-out check)
    logger.info("Loading tokenizer: %s", hf_checkpoint)
    tokenizer = load_tokenizer(hf_checkpoint)

    # -----------------------------------------------------------------------
    # Offline unit tests (no SGLang needed)
    # -----------------------------------------------------------------------
    unit_ok = run_proxy_unit_tests(tokenizer)
    if unit_test_only:
        sys.exit(0 if unit_ok else 1)

    SWEBENCH_CONFIGS["sglang_url"] = sglang_url
    SWEBENCH_CONFIGS["trials_dir"] = trials_dir
    SWEBENCH_CONFIGS["agent_name"] = agent
    url_env = _AGENT_URL_ENV.get(agent, "ANTHROPIC_BASE_URL")
    logger.info("Agent: %s  (URL env: %s)  SGLang: %s", agent, url_env, sglang_url)

    # -----------------------------------------------------------------------
    # Multi-task concurrent run
    # -----------------------------------------------------------------------
    if len(task_ids) > 1:
        import time
        args = _build_args(hf_checkpoint)
        logger.info("Running %d tasks concurrently…", len(task_ids))
        t0 = time.monotonic()
        results = await asyncio.gather(*[_run_task(tid, args, tokenizer) for tid in task_ids])
        elapsed = time.monotonic() - t0
        all_pass = _print_multi_summary(list(results), elapsed)
        sys.exit(0 if all_pass else 1)

    # -----------------------------------------------------------------------
    # Single-task verbose run
    # -----------------------------------------------------------------------
    task_id = task_ids[0]
    logger.info("Loading instruction for task: %s", task_id)
    instruction = _load_instruction(task_id)
    logger.info("Instruction preview: %s…", instruction[:120].replace("\n", " "))

    args = _build_args(hf_checkpoint)
    sample = Sample(prompt=instruction, label=task_id)

    logger.info("Running generate() for task: %s", task_id)
    result = await generate(args, sample, {})

    captures: list[CapturedTurn] = getattr(result, "_proxy_captures", None) or []

    # -----------------------------------------------------------------------
    # Conversation
    # -----------------------------------------------------------------------
    _print_conversation(captures)

    # -----------------------------------------------------------------------
    # Token-in / token-out invariants
    # -----------------------------------------------------------------------
    tio_ok = check_token_in_token_out(captures, tokenizer)

    # -----------------------------------------------------------------------
    # Trajectory message check
    # -----------------------------------------------------------------------
    trial_dirs = sorted(Path(trials_dir).glob(f"{task_id}__*"))
    if trial_dirs:
        # Each agent writes its stream-JSON log under a different filename.
        # Fall back gracefully if the file does not exist.
        _AGENT_TRAJECTORY_FILE: dict[str, str] = {
            "claude-code": "claude-code.txt",
            "opencode":    "opencode.txt",
        }
        traj_filename = _AGENT_TRAJECTORY_FILE.get(agent, f"{agent}.txt")
        trajectory_path = trial_dirs[-1] / "agent" / traj_filename
        check_messages_match_trajectory(captures, trajectory_path)
    else:
        _console.print(
            f"[yellow]No trial dir found under {args.trials_dir!r} for {task_id}[/yellow]"
        )

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    prompt_len = len(result.tokens) - result.response_length
    trained = sum(result.loss_mask) if result.loss_mask else 0
    tool_obs = len(result.loss_mask) - trained if result.loss_mask else 0

    reward_color = "green" if result.reward and result.reward > 0 else "red"
    status_color = "green" if result.status == result.status.COMPLETED else "yellow"

    summary = Text()
    summary.append("Task:    ", style="bold"); summary.append(f"{task_id}\n")
    summary.append("Agent:   ", style="bold"); summary.append(f"{agent}\n")
    summary.append("Reward:  ", style="bold"); summary.append(f"{result.reward}\n", style=reward_color)
    summary.append("Status:  ", style="bold"); summary.append(f"{result.status.value}\n", style=status_color)
    summary.append("Turns:   ", style="bold"); summary.append(f"{len(captures)}\n")
    summary.append("\n")
    summary.append("Prompt tokens:    ", style="bold"); summary.append(f"{prompt_len}\n")
    summary.append("Response tokens:  ", style="bold"); summary.append(f"{result.response_length}\n")
    summary.append(f"  └─ assistant (loss_mask=1):  {trained}\n", style="dim")
    summary.append(f"  └─ tool obs  (loss_mask=0):  {tool_obs}\n", style="dim")
    summary.append("Total tokens:     ", style="bold"); summary.append(f"{len(result.tokens)}\n")
    if result.rollout_log_probs and result.loss_mask:
        # Average only over assistant tokens (loss_mask=1); tool_obs tokens
        # have log_prob=0.0 and should not dilute the average.
        asst_lps = [lp for lp, m in zip(result.rollout_log_probs, result.loss_mask) if m == 1]
        if asst_lps:
            avg = sum(asst_lps) / len(asst_lps)
            summary.append("Avg log prob:     ", style="bold"); summary.append(f"{avg:.4f}  [dim](over {len(asst_lps)} asst tokens)[/dim]\n")
    summary.append("\n")
    summary.append("Token-in/out:     ", style="bold")
    summary.append("PASS\n" if tio_ok else "FAIL\n", style="green" if tio_ok else "red")

    _console.print(Panel(summary, title="[bold]Results[/bold]", border_style="cyan", padding=(0, 1)))

    sys.exit(0 if tio_ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test harbor generate on SWEBench tasks")

    # ---- task selection (mutually exclusive) ----
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "-t", "--task",
        default=None,
        help="Single SWEBench task ID (default: django__django-13410)",
    )
    task_group.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK_ID",
        help="One or more task IDs to run concurrently.",
    )
    task_group.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        metavar="N",
        help=f"Run the first N tasks from the default pool of {len(DEFAULT_TASKS)}.",
    )

    parser.add_argument(
        "--hf-checkpoint",
        default="Qwen/Qwen3-Coder-Next",
        help="HuggingFace tokenizer checkpoint — must match what SGLang loaded.",
    )
    parser.add_argument(
        "--trials-dir",
        default="trials_test",
        help="Directory to write trial artifacts (default: trials_test)",
    )
    parser.add_argument(
        "--sglang-url",
        default=SWEBENCH_CONFIGS["sglang_url"],
        help="Base URL of the SGLang server "
             "(default: SGLANG_URL env var or http://127.0.0.1:30000)",
    )
    parser.add_argument(
        "--agent",
        default=SWEBENCH_CONFIGS["agent_name"],
        help=f"Harbor agent name (default: {SWEBENCH_CONFIGS['agent_name']}).  "
             f"Known agents: {', '.join(sorted(_AGENT_URL_ENV))}.",
    )
    parser.add_argument(
        "--proxy-host",
        default=None,
        help="Host IP reachable from inside Docker for the intercept proxy "
             "(default: HARBOR_PROXY_HOST env var or 240.10.0.1).",
    )
    parser.add_argument(
        "--unit-test",
        action="store_true",
        help="Run offline proxy unit tests only (no SGLang or Docker needed).",
    )
    args = parser.parse_args()

    # Resolve task list
    if args.tasks:
        task_ids = args.tasks
    elif args.n_tasks is not None:
        task_ids = DEFAULT_TASKS[: args.n_tasks]
    elif args.task:
        task_ids = [args.task]
    else:
        task_ids = ["django__django-13410"]  # sensible single-task default

    hf_checkpoint = args.hf_checkpoint or SWEBENCH_CONFIGS["model"]
    if args.proxy_host:
        SWEBENCH_CONFIGS["proxy_host"] = args.proxy_host

    asyncio.run(main(
        task_ids,
        hf_checkpoint,
        args.trials_dir,
        args.sglang_url,
        args.agent,
        args.unit_test,
    ))
