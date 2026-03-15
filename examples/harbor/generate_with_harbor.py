"""Harbor rollout using a black-box agent on SWEBench tasks via harbor's Trial API.

Runs harbor's Trial directly (Docker env + agent + verifier) while intercepting
LLM calls via ModelInterceptProxy to capture exact token-level trajectory data
for RL training.

Supports two execution modes:
- Local (single node):
    --custom-generate-function-path examples.harbor.generate_with_harbor.generate

- Distributed (multi-node with Ray):
    --custom-generate-function-path examples.harbor.generate_with_harbor.distributed_generate

Sample metadata keys:
    - agent_name: str - Harbor agent to use (default: "claude-code")
    - model_name: str - Model name passed to the harbor agent
    - agent_timeout_s: float - Agent execution timeout in seconds (default: 1800)
    - agent_env: dict - Additional env vars forwarded to the agent
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from slime.utils.types import Sample
from transformers import AutoTokenizer

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial

from autonomy.rollouts.distributed_utils import make_distributed
from autonomy.rollouts.generate_interface import SamplingParams, typed_generate
from autonomy.utils.rollout_utils import track_time
from autonomy.utils.types import SlimeArgs

from model_proxy import CapturedTurn, ModelInterceptProxy

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Each agent reads its API base URL from a different environment variable.
# The proxy binds per-trial and injects the right var so the agent's traffic
# flows through it.
_AGENT_URL_ENV: dict[str, str] = {
    "claude-code":    "ANTHROPIC_BASE_URL",   # Anthropic /v1/messages
    "opencode":       "ANTHROPIC_BASE_URL",   # Anthropic /v1/messages
    "aider":          "OPENAI_BASE_URL",      # OpenAI /v1/chat/completions
    "goose":          "OPENAI_BASE_URL",
    "swe-agent":      "OPENAI_BASE_URL",
    "codex":          "OPENAI_BASE_URL",
    "kimi-cli":       "OPENAI_BASE_URL",
    "qwen-code":      "OPENAI_BASE_URL",
    "mini-swe-agent": "OPENAI_API_BASE",
    "openhands":      "LLM_BASE_URL",
    "openhands-sdk":  "LLM_BASE_URL",
}

# Default configuration — overridden via sample.metadata or environment variables.
SWEBENCH_CONFIGS: dict[str, Any] = {
    "agent_name": "claude-code",
    "model": "qwen3-coder",
    "max_turns": 100,
    "dataset_path_prefix": "datasets/swebench-verified",
    "git_url": "https://github.com/laude-institute/harbor-datasets.git",
    "git_commit_id": "86723674f04e4209ac479d0fb75d9d9f44b4377e",
    "trials_dir": os.environ.get("HARBOR_TRIALS_DIR", "trials"),
    # SGLang server — proxy calls /generate directly for exact token-in/token-out
    "sglang_url": os.environ.get("SGLANG_URL", "http://127.0.0.1:30000"),
    "agent_env": {
        "ANTHROPIC_AUTH_TOKEN": os.environ.get("ANTHROPIC_AUTH_TOKEN", "sk-1234"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        # ANTHROPIC_BASE_URL is set per-trial to point at the proxy
    },
    "n_concurrent_tasks": int(os.environ.get("HARBOR_N_CONCURRENT", "16")),
    # Host IP reachable from inside Docker containers.
    # Harbor assigns 240.10.0.1 as the docker-bridge gateway; override with
    # HARBOR_PROXY_HOST if your network differs.
    "proxy_host": os.environ.get("HARBOR_PROXY_HOST", "172.17.0.1"),
    "proxy_port_base": int(os.environ.get("HARBOR_PROXY_PORT_BASE", "19000")),
}


# =============================================================================
# Port pool
# =============================================================================

_port_pool: asyncio.Queue | None = None


async def _get_port_pool() -> asyncio.Queue:
    global _port_pool
    if _port_pool is None:
        _port_pool = asyncio.Queue()
        base = SWEBENCH_CONFIGS["proxy_port_base"]
        for i in range(SWEBENCH_CONFIGS["n_concurrent_tasks"]):
            await _port_pool.put(base + i)
    return _port_pool


# =============================================================================
# TrialConfig builder
# =============================================================================


def _build_trial_config(
    task_id: str,
    task_dir: str | Path | None,
    trials_dir: str | Path,
    proxy_port: int,
    agent_name: str,
    model_name: str,
    max_turns: int,
    agent_env: dict[str, str],
) -> TrialConfig:
    """Build a TrialConfig pointing the agent's API URL env var at the per-trial proxy.

    When *task_dir* is provided (from ``sample.metadata["instance"]["task_dir"]``),
    the task is loaded from the local filesystem with no git clone.
    Otherwise falls back to downloading from the configured git repository.
    """
    cfg = SWEBENCH_CONFIGS
    proxy_url = f"http://{cfg['proxy_host']}:{proxy_port}"

    if task_dir is not None:
        task = TaskConfig(
            path=Path(task_dir),
            source="swebench-verified",
        )
    else:
        task = TaskConfig(
            path=Path(cfg["dataset_path_prefix"]) / task_id,
            git_url=cfg["git_url"],
            git_commit_id=cfg["git_commit_id"],
            source="swebench-verified",
        )

    env = dict(agent_env)
    url_env_var = _AGENT_URL_ENV.get(agent_name, "ANTHROPIC_BASE_URL")
    env[url_env_var] = proxy_url

    agent = AgentConfig(
        name=agent_name,
        model_name=model_name,
        kwargs={"max_turns": max_turns},
        env=env,
    )

    return TrialConfig(
        task=task,
        agent=agent,
        environment=EnvironmentConfig(),
        verifier=VerifierConfig(),
        trials_dir=Path(trials_dir),
        job_id=uuid.uuid4(),
    )


# =============================================================================
# Token array construction from proxy captures
# =============================================================================


def _build_token_arrays(
    captures: list[CapturedTurn],
) -> tuple[list[int], list[int], list[float]]:
    """Build full token IDs, response loss mask, and response log probs.

    This function is a **trivial concatenation** — it performs no tokenizer
    calls and no approximations.  Correctness relies entirely on the proxy's
    ``_accumulated_ids`` invariant:

        captures[i+1].input_ids  ==  captures[i].input_ids
                                    + captures[i].output_ids   (exact, from SGLang)
                                    + tool_obs[i]              (delta from stub method)

    Given that invariant, the tool-observation slice requires no decoding::

        tool_obs[i] = captures[i+1].input_ids[ len(all_ids so far) : ]

    Token layout::

        all_ids = input_ids[0]           ← prompt            (loss_mask = 0)
                + output_ids[0]          ← assistant turn 0  (loss_mask = 1)
                + tool_obs[0]            ← tool results + next gen-prompt
                                                              (loss_mask = 0)
                + output_ids[1]          ← assistant turn 1  (loss_mask = 1)
                + …
                + output_ids[N]          ← final assistant   (loss_mask = 1)

    Returns:
        (all_ids, response_loss_mask, response_log_probs)
        ``all_ids`` spans prompt + full response.
        ``response_loss_mask`` and ``response_log_probs`` span the response
        portion only (length = ``len(all_ids) - len(prompt_ids)``).
    """
    if not captures:
        return [], [], []

    prompt_ids = captures[0].input_ids
    all_ids: list[int] = list(prompt_ids)
    loss_mask: list[int] = [0] * len(prompt_ids)
    response_log_probs: list[float] = []

    for i, turn in enumerate(captures):
        # Assistant tokens — train on these
        all_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        response_log_probs.extend(turn.log_probs)

        # Tool observation tokens + next generation prompt — do not train
        if i + 1 < len(captures):
            tool_obs = captures[i + 1].input_ids[len(all_ids):]
            all_ids.extend(tool_obs)
            loss_mask.extend([0] * len(tool_obs))
            response_log_probs.extend([0.0] * len(tool_obs))

    response_loss_mask = loss_mask[len(prompt_ids):]
    return all_ids, response_loss_mask, response_log_probs


# =============================================================================
# Logging helper
# =============================================================================


def _populate_logging_info(
    sample: Sample,
    rollout_timing: dict[str, list[float]],
    failure_stage: str | None,
    reward: float,
    n_turns: int,
    n_prompt_tokens: int,
    n_response_tokens: int,
) -> None:
    """Populate sample.metadata['logging_info'] for downstream logging."""
    sample.metadata["logging_info"] = {
        "rollout_timing": rollout_timing,
        "failure_stage": failure_stage,
        "env_reward": reward,
        "n_turns": n_turns,
        "n_prompt_tokens": n_prompt_tokens,
        "n_response_tokens": n_response_tokens,
    }


# =============================================================================
# Generate
# =============================================================================


@typed_generate
async def generate(
    args: SlimeArgs,
    sample: Sample,
    sampling_params: SamplingParams,
    evaluation: bool = False,
) -> Sample:
    """SWEBench rollout via harbor Trial + black-box agent + SGLang proxy.

    Lifecycle:
    1. Read agent configuration from sample.metadata (with SWEBENCH_CONFIGS defaults)
    2. Setup tokenizer
    3. Acquire proxy port from pool
    4. Start ModelInterceptProxy
    5. Build TrialConfig pointing the agent's URL env var at the proxy
    6. Run Trial (Docker env setup → agent → verifier)
    7. Stop proxy and release port
    8. Build token arrays from captures (trivial: no tokenizer call needed)
    9. Return populated Sample

    Args:
        args:            SlimeArgs with model checkpoint and SGLang router config.
        sample:          sample.label = SWEBench task ID.
        sampling_params: Generation parameters forwarded to SGLang /generate.
        evaluation:      Whether this is an evaluation-only rollout.

    Returns:
        Populated Sample with exact token IDs, loss mask, and log probs.
    """
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}

    # -- 1. Read agent configuration from metadata --
    meta = sample.metadata
    agent_name = meta.get("agent_name") or SWEBENCH_CONFIGS["agent_name"]
    model_name = meta.get("model_name") or args.model_name or SWEBENCH_CONFIGS["model"]
    agent_timeout_s = float(meta.get("agent_timeout_s", 1800))
    agent_env = {**SWEBENCH_CONFIGS["agent_env"], **meta.get("agent_env", {})}
    trials_dir = meta.get("trials_dir") or SWEBENCH_CONFIGS["trials_dir"]

    # task_id is passed as sample.prompt (--input-key task_id in train.sh).
    # Fall back to sample.label for backward compatibility.
    task_id: str = sample.prompt or sample.label or ""
    if not task_id:
        logger.error("task_id is empty — sample.prompt and sample.label are both unset")
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return sample

    # Local task directory from parquet metadata — avoids per-trial git clone.
    task_dir: str | None = meta.get("instance", {}).get("task_dir")

    # -- 2. Setup tokenizer and SGLang endpoint --
    model_path = args.hf_checkpoint or os.environ.get("MODEL_PATH", "")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use the SGLang router endpoint managed by slime (args.sglang_router_ip/port),
    # falling back to SWEBENCH_CONFIGS["sglang_url"] for standalone / debug runs.
    if args.sglang_router_ip and args.sglang_router_port:
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    else:
        sglang_url = SWEBENCH_CONFIGS["sglang_url"]

    # Build SGLang /generate sampling parameters from the incoming SamplingParams.
    sglang_params: dict[str, Any] = {
        "max_new_tokens": sampling_params.max_new_tokens or args.rollout_max_response_len or 32768,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "skip_special_tokens": False,
    }

    # -- Timing and state tracking --
    rollout_timing: dict[str, list[float]] = {
        "proxy_start": [],
        "agent_run": [],
        "cleanup": [],
    }
    failure_stage: str | None = None
    proxy: ModelInterceptProxy | None = None
    trial_result = None

    # -- 3. Acquire proxy port --
    port_pool = await _get_port_pool()
    proxy_port: int = await port_pool.get()
    reward: float = 0.0

    try:
        # -- 4. Start proxy --
        async with track_time(rollout_timing, "proxy_start"):
            proxy = ModelInterceptProxy(
                port=proxy_port,
                sglang_url=sglang_url,
                tokenizer=tokenizer,
                sampling_params=sglang_params,
            )
            try:
                await proxy.start()
            except Exception:
                failure_stage = "proxy_start"
                raise

        # -- 5. Build TrialConfig --
        config = _build_trial_config(
            task_id=task_id,
            task_dir=task_dir,
            trials_dir=trials_dir,
            proxy_port=proxy_port,
            agent_name=agent_name,
            model_name=model_name,
            max_turns=SWEBENCH_CONFIGS["max_turns"],
            agent_env=agent_env,
        )

        # -- 6. Run Trial --
        async with track_time(rollout_timing, "agent_run"):
            try:
                trial = Trial(config)
                trial_result = await asyncio.wait_for(trial.run(), timeout=agent_timeout_s)
                reward = _extract_reward(trial_result)
            except asyncio.TimeoutError:
                logger.warning("Harbor trial timed out after %.0fs: task=%s", agent_timeout_s, task_id)
                failure_stage = "agent_run_timeout"
            except Exception as exc:
                logger.error("Harbor trial error task=%s: %s", task_id, exc, exc_info=True)
                failure_stage = "agent_run"

    except Exception as exc:
        logger.error("Harbor rollout error: %s", exc, exc_info=True)
        if failure_stage is None:
            failure_stage = "unknown"
    finally:
        # -- 7. Stop proxy and release port --
        async with track_time(rollout_timing, "cleanup"):
            if proxy is not None:
                await proxy.stop()
            await port_pool.put(proxy_port)

    # -- 8. Build token arrays from proxy captures --
    captures: list[CapturedTurn] = proxy.captures if proxy is not None else []

    if not captures:
        logger.warning("No proxy captures for task %s (failure_stage=%s)", task_id, failure_stage)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        prompt_text = sample.prompt or ""
        prompt_ids: list[int] = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_ids + [pad_id]
        sample.prompt = prompt_text
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = [0.0]
        sample.reward = reward
        sample.remove_sample = True
        sample.status = Sample.Status.FAILED

        _populate_logging_info(sample, rollout_timing, failure_stage or "empty_captures",
                               reward, 0, len(prompt_ids), 0)
        return sample

    all_token_ids, response_loss_mask, response_log_probs = _build_token_arrays(captures)

    prompt_token_ids = captures[0].input_ids
    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    response_token_ids = all_token_ids[len(prompt_token_ids):]

    # -- 9. Determine status and populate Sample --
    exc_info = getattr(trial_result, "exception_info", None) if trial_result is not None else None
    if failure_stage == "agent_run_timeout" or (
        exc_info is not None and "Timeout" in getattr(exc_info, "exception_type", "")
    ):
        sample.status = Sample.Status.TRUNCATED
    elif failure_stage is not None:
        sample.status = Sample.Status.FAILED
    else:
        sample.status = Sample.Status.COMPLETED

    n_asst_tokens = sum(response_loss_mask)
    n_tool_tokens = len(response_loss_mask) - n_asst_tokens

    logger.info(
        "Harbor rollout done | task=%s reward=%.1f turns=%d "
        "prompt=%d response=%d (asst=%d tool_obs=%d)",
        task_id, reward, len(captures),
        len(prompt_token_ids), len(response_token_ids),
        n_asst_tokens, n_tool_tokens,
    )

    sample.tokens = all_token_ids
    sample.prompt = prompt_text
    sample.response_length = len(response_token_ids)
    sample.response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    sample.loss_mask = response_loss_mask
    sample.rollout_log_probs = response_log_probs
    sample.reward = reward
    sample._proxy_captures = captures  # type: ignore[attr-defined]

    _populate_logging_info(sample, rollout_timing, failure_stage,
                           reward, len(captures), len(prompt_token_ids), len(response_token_ids))

    return sample


# =============================================================================
# Helper Functions
# =============================================================================


def _extract_reward(trial_result: Any) -> float:
    """Extract scalar reward from harbor trial result."""
    vr = getattr(trial_result, "verifier_result", None)
    if vr is None:
        return 0.0
    rewards: dict[str, Any] | None = getattr(vr, "rewards", None)
    if not rewards:
        return 0.0
    if "reward" in rewards:
        return float(rewards["reward"])
    numeric = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


# =============================================================================
# Reward function
# =============================================================================


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Return the reward already stored by the verifier during generate()."""
    return float(getattr(sample, "reward", None) or 0.0)


# =============================================================================
# Distributed Entry Point
# =============================================================================

_TASK_NUM_CPUS = float(os.environ.get("SLIME_DISTRIBUTED_TASK_CPUS", "2.0"))

distributed_generate = make_distributed(generate, wrap_list=True, default_num_cpus=_TASK_NUM_CPUS)
