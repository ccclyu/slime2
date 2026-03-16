"""Run Harbor trials behind the intercept proxy and return exact token trajectories."""

import asyncio
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
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

from autonomy.data.rl.agent.harbor.build_harbor import unpack_task_files
from autonomy.rollouts.distributed_utils import make_distributed
from autonomy.rollouts.generate_interface import SamplingParams, typed_generate
from autonomy.utils.rollout_utils import track_time
from autonomy.utils.types import SlimeArgs

from examples.harbor.model_proxy import AgentTurn, ModelInterceptProxy

logger = logging.getLogger(__name__)


@dataclass
class HarborRunConfig:
    task_id: str
    task_dir: str
    task_source: str
    agent_name: str
    agent_url_env: str
    model_name: str
    max_turns: int
    agent_timeout_s: float
    agent_env: dict[str, str]
    trials_dir: str
    proxy_host: str | None
    proxy_port_base: int
    n_concurrent_tasks: int


@dataclass
class TrialRunResult:
    turns: list[AgentTurn]
    reward: float
    failure_stage: str | None
    rollout_timing: dict[str, list[float]]
    trial_result: Any
    tokenizer: Any
def _require_arg(args: SlimeArgs, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None:
        raise ValueError(f"args.{name} must be set, typically via --custom-config-path")
    return value


def _resolve_proxy_host(configured_host: str | None = None) -> str:
    """Resolve the host IP reachable from inside Docker containers on this node."""
    if configured_host:
        return configured_host
    try:
        result = subprocess.run(
            ["ip", "route", "show", "dev", "docker0"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.splitlines():
            if "src" in line:
                return line.split("src")[1].strip().split()[0]
    except Exception:
        pass
    return "172.17.0.1"

_port_pool: asyncio.Queue | None = None
_port_pool_config: tuple[int, int] | None = None

async def _get_port_pool(proxy_port_base: int, n_concurrent_tasks: int) -> asyncio.Queue:
    global _port_pool, _port_pool_config
    desired_config = (proxy_port_base, n_concurrent_tasks)
    if _port_pool is None or _port_pool_config != desired_config:
        _port_pool = asyncio.Queue()
        for i in range(n_concurrent_tasks):
            await _port_pool.put(proxy_port_base + i)
        _port_pool_config = desired_config
    return _port_pool

def _build_trial_config(
    task_dir: str | Path,
    task_source: str,
    trials_dir: str | Path,
    proxy_port: int,
    agent_name: str,
    agent_url_env: str,
    model_name: str,
    max_turns: int,
    agent_env: dict[str, str],
    proxy_host: str | None,
) -> TrialConfig:
    """Build a TrialConfig pointing the agent's API URL env var at the proxy."""
    proxy_url = f"http://{_resolve_proxy_host(proxy_host)}:{proxy_port}"

    task = TaskConfig(
        path=Path(task_dir),
        source=task_source,
    )

    env = {**agent_env, agent_url_env: proxy_url}

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


def _build_token_arrays(
    turns: list[AgentTurn],
) -> tuple[list[int], list[int], list[float]]:
    """Build token IDs, response loss mask, and response log probs from captured turns."""
    if not turns:
        return [], [], []

    prompt_ids = turns[0].input_ids
    all_ids: list[int] = list(prompt_ids)
    loss_mask: list[int] = [0] * len(prompt_ids)
    response_log_probs: list[float] = []

    for i, turn in enumerate[AgentTurn](turns):
        all_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))
        response_log_probs.extend(turn.log_probs)

        if i + 1 < len(turns):
            tool_obs = turns[i + 1].input_ids[len(all_ids):]
            all_ids.extend(tool_obs)
            loss_mask.extend([0] * len(tool_obs))
            response_log_probs.extend([0.0] * len(tool_obs))

    response_loss_mask = loss_mask[len(prompt_ids):]
    return all_ids, response_loss_mask, response_log_probs


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


def _build_run_config(args: SlimeArgs, sample: Sample) -> HarborRunConfig:
    """Resolve task fields from sample metadata and Harbor runtime settings from args."""
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}

    meta = sample.metadata
    agent_name = str(_require_arg(args, "agent_name"))
    agent_url_env = str(_require_arg(args, "agent_url_env"))
    model_name: str = args.model_name or args.hf_checkpoint or ""
    if not model_name:
        raise ValueError("model_name must be set via args.model_name or args.hf_checkpoint")

    task_id: str = meta.get("instance", {}).get("task_id") or sample.label or ""
    if not task_id:
        raise ValueError("task_id is empty — metadata.instance.task_id and sample.label are both unset")

    task_dir = meta.get("instance", {}).get("task_dir")
    task_files: str | None = meta.get("instance", {}).get("task_files")
    if task_dir is not None and not Path(task_dir).exists() and task_files:
        unpack_task_files(task_files, Path(task_dir))
        logger.info("Unpacked task_files to %s", task_dir)
    if task_dir is None:
        raise ValueError(f"task_id={task_id}: metadata.instance.task_dir is missing and no task_files to unpack")

    return HarborRunConfig(
        task_id=task_id,
        task_dir=task_dir,
        task_source=meta.get("data_source", "harbor"),
        agent_name=agent_name,
        agent_url_env=agent_url_env,
        model_name=model_name,
        max_turns=int(_require_arg(args, "rollout_max_turns")),
        agent_timeout_s=float(_require_arg(args, "agent_timeout_s")),
        agent_env={
            **(getattr(args, "harbor_agent_env", None) or {}),
        },
        trials_dir=str(_require_arg(args, "trials_dir")),
        proxy_host=getattr(args, "harbor_proxy_host", None),
        proxy_port_base=int(_require_arg(args, "harbor_proxy_port_base")),
        n_concurrent_tasks=int(_require_arg(args, "harbor_n_concurrent_tasks")),
    )


def _build_runtime(
    args: SlimeArgs,
    sampling_params: SamplingParams,
) -> tuple[Any, str, dict[str, Any]]:
    """Build tokenizer and SGLang runtime settings for one rollout."""
    model_path = args.hf_checkpoint or os.environ.get("MODEL_PATH", "")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not (args.sglang_router_ip and args.sglang_router_port):
        raise ValueError("args.sglang_router_ip and args.sglang_router_port must be set")
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

    sglang_params: dict[str, Any] = {
        "max_new_tokens": sampling_params.max_new_tokens or args.rollout_max_response_len or 32768,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "skip_special_tokens": False,
    }
    return tokenizer, sglang_url, sglang_params


async def _run_harbor_trial(
    run_config: HarborRunConfig,
    tokenizer: Any,
    sglang_url: str,
    sglang_params: dict[str, Any],
) -> TrialRunResult:
    """Run one Harbor trial behind the intercept proxy and capture rollout metadata."""
    rollout_timing: dict[str, list[float]] = {
        "proxy_start": [],
        "agent_run": [],
        "cleanup": [],
    }
    failure_stage: str | None = None
    trial_result = None
    reward = 0.0

    port_pool = await _get_port_pool(run_config.proxy_port_base, run_config.n_concurrent_tasks)
    proxy_port: int = await port_pool.get()
    proxy: ModelInterceptProxy | None = None

    try:
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

        config = _build_trial_config(
            task_dir=run_config.task_dir,
            task_source=run_config.task_source,
            trials_dir=run_config.trials_dir,
            proxy_port=proxy_port,
            agent_name=run_config.agent_name,
            agent_url_env=run_config.agent_url_env,
            model_name=run_config.model_name,
            max_turns=run_config.max_turns,
            agent_env=run_config.agent_env,
            proxy_host=run_config.proxy_host,
        )

        async with track_time(rollout_timing, "agent_run"):
            try:
                trial = Trial(config)
                trial_result = await asyncio.wait_for(trial.run(), timeout=run_config.agent_timeout_s)
                reward = _extract_reward(trial_result)
            except asyncio.TimeoutError:
                logger.warning(
                    "Harbor trial timed out after %.0fs: task=%s",
                    run_config.agent_timeout_s,
                    run_config.task_id,
                )
                failure_stage = "agent_run_timeout"
            except Exception as exc:
                logger.error("Harbor trial error task=%s: %s", run_config.task_id, exc, exc_info=True)
                failure_stage = "agent_run"

    except Exception as exc:
        logger.error("Harbor rollout error: %s", exc, exc_info=True)
        if failure_stage is None:
            failure_stage = "unknown"
    finally:
        async with track_time(rollout_timing, "cleanup"):
            if proxy is not None:
                await proxy.stop()
            await port_pool.put(proxy_port)

    return TrialRunResult(
        turns=proxy.turns if proxy is not None else [],
        reward=reward,
        failure_stage=failure_stage,
        rollout_timing=rollout_timing,
        trial_result=trial_result,
        tokenizer=tokenizer,
    )


def _resolve_sample_status(failure_stage: str | None, trial_result: Any) -> Sample.Status:
    exc_info = getattr(trial_result, "exception_info", None) if trial_result is not None else None
    if failure_stage == "agent_run_timeout" or (
        exc_info is not None and "Timeout" in getattr(exc_info, "exception_type", "")
    ):
        return Sample.Status.TRUNCATED
    if failure_stage is not None:
        return Sample.Status.FAILED
    return Sample.Status.COMPLETED


def _populate_empty_turn_sample(
    sample: Sample,
    tokenizer: Any,
    reward: float,
    failure_stage: str | None,
    rollout_timing: dict[str, list[float]],
) -> Sample:
    """Populate a failed sample when the proxy captured no turns."""
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

    _populate_logging_info(sample, rollout_timing, failure_stage or "empty_turns",
                           reward, 0, len(prompt_ids), 0)
    return sample


def _populate_turn_sample(
    sample: Sample,
    trial_run: TrialRunResult,
) -> Sample:
    """Populate a sample from captured proxy turns."""
    turns = trial_run.turns
    all_token_ids, response_loss_mask, response_log_probs = _build_token_arrays(turns)

    prompt_token_ids = turns[0].input_ids
    prompt_text = trial_run.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    response_token_ids = all_token_ids[len(prompt_token_ids):]
    sample.status = _resolve_sample_status(trial_run.failure_stage, trial_run.trial_result)

    n_asst_tokens = sum(response_loss_mask)
    n_tool_tokens = len(response_loss_mask) - n_asst_tokens
    logger.info(
        "Harbor rollout done | task=%s reward=%.1f turns=%d "
        "prompt=%d response=%d (asst=%d tool_obs=%d)",
        sample.label or "",
        trial_run.reward,
        len(turns),
        len(prompt_token_ids),
        len(response_token_ids),
        n_asst_tokens,
        n_tool_tokens,
    )

    sample.tokens = all_token_ids
    sample.prompt = prompt_text
    sample.response_length = len(response_token_ids)
    sample.response = trial_run.tokenizer.decode(response_token_ids, skip_special_tokens=False)
    sample.loss_mask = response_loss_mask
    sample.rollout_log_probs = response_log_probs
    sample.reward = trial_run.reward
    sample._proxy_turns = turns  # type: ignore[attr-defined]

    _populate_logging_info(
        sample,
        trial_run.rollout_timing,
        trial_run.failure_stage,
        trial_run.reward,
        len(turns),
        len(prompt_token_ids),
        len(response_token_ids),
    )
    return sample


@typed_generate
async def generate(
    args: SlimeArgs,
    sample: Sample,
    sampling_params: SamplingParams,
    evaluation: bool = False,
) -> Sample:
    """Run one Harbor rollout and populate the sample from captured proxy turns."""
    del evaluation
    try:
        run_config = _build_run_config(args, sample)
    except Exception as exc:
        logger.error("Failed to build Harbor run config: %s", exc, exc_info=True)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return sample

    tokenizer, sglang_url, sglang_params = _build_runtime(args, sampling_params)
    trial_run = await _run_harbor_trial(run_config, tokenizer, sglang_url, sglang_params)

    if not trial_run.turns:
        logger.warning(
            "No proxy turns for task %s (failure_stage=%s)",
            run_config.task_id,
            trial_run.failure_stage,
        )
        return _populate_empty_turn_sample(
            sample,
            trial_run.tokenizer,
            trial_run.reward,
            trial_run.failure_stage,
            trial_run.rollout_timing,
        )

    return _populate_turn_sample(sample, trial_run)

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

async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Return the reward already stored by the verifier during generate()."""
    del args, kwargs
    return float(getattr(sample, "reward", None) or 0.0)

_TASK_NUM_CPUS = float(os.environ.get("SLIME_DISTRIBUTED_TASK_CPUS", "2.0"))

distributed_generate = make_distributed(generate, wrap_list=True, default_num_cpus=_TASK_NUM_CPUS)
