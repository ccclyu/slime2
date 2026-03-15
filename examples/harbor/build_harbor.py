#!/usr/bin/env python3
"""Build a SLIME-compatible parquet for harbor rollout from SWEBench verified tasks.

Each parquet row contains only the task_id (no instruction text). The local
task directory path is stored in metadata so generate_with_harbor.py can
pass it directly to harbor's Trial at rollout time — no git clone per trial.

Parquet schema::

    task_id  (str)  — e.g. "django__django-12345"  → loaded as sample.prompt
    metadata (dict) — {
        "instance": {
            "task_id":  str,
            "task_dir": str,   # absolute local path used by Trial at rollout time
        }
    }

Usage::

    # Download all tasks (lists them via git ls-tree first, then downloads)
    python build_harbor.py --download

    # Download specific tasks only
    python build_harbor.py --download --tasks django__django-13410 sympy__sympy-19346

    # Point at an already-downloaded task directory
    python build_harbor.py --task-dir ~/.cache/harbor/tasks

    # Override output path
    python build_harbor.py --download --output /custom/path/train.parquet
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from harbor.models.task.id import GitTaskId
from harbor.tasks.client import TaskClient

logger = logging.getLogger(__name__)

# SWEBench verified dataset location in the harbor-datasets git repo.
_GIT_URL = "https://github.com/laude-institute/harbor-datasets.git"
_GIT_COMMIT = "86723674f04e4209ac479d0fb75d9d9f44b4377e"
_DATASET_PREFIX = "datasets/swebench-verified"

# Instruction file names tried in order.
_INSTRUCTION_FILES: tuple[str, ...] = (
    "instruction.md",
    "instruction.txt",
    "instructions.txt",
    "README.md",
)


# =============================================================================
# Task downloader
# =============================================================================


def _list_all_task_names() -> list[str]:
    """List all task directory names under _DATASET_PREFIX at _GIT_COMMIT.

    Uses a filterless shallow clone + ``git ls-tree`` so no blob data is
    transferred — only the tree object is fetched.

    Returns:
        Sorted list of task directory names, e.g. ["django__django-10097", …].
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
             _GIT_URL, str(tmp_path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", _GIT_COMMIT],
            check=True, capture_output=True, cwd=tmp_path,
        )
        result = subprocess.run(
            ["git", "ls-tree", "--name-only", _GIT_COMMIT, _DATASET_PREFIX + "/"],
            check=True, capture_output=True, text=True, cwd=tmp_path,
        )

    # git ls-tree returns full paths like "datasets/swebench-verified/django__django-12345"
    # — take only the basename (the actual task directory name).
    names = [Path(line.strip()).name for line in result.stdout.splitlines() if line.strip()]
    logger.info("Found %d tasks in %s @ %s", len(names), _DATASET_PREFIX, _GIT_COMMIT[:12])
    return sorted(names)


def download_tasks(
    task_names: list[str] | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download SWEBench verified tasks via harbor's TaskClient.

    Tasks are fetched from the harbor-datasets git repo and cached under
    ``~/.cache/harbor/tasks/`` (or *output_dir* if given).

    Args:
        task_names: Task IDs to download. Pass ``None`` to download all tasks
            (lists them via ``git ls-tree`` first, then downloads in one batch).
        output_dir: Override harbor's default cache directory.
        overwrite:  Re-download even if already cached.

    Returns:
        List of local task directory paths, one per task.
    """
    if task_names is None:
        logger.info("No --tasks specified — listing all tasks from git repo …")
        task_names = _list_all_task_names()

    task_ids = [
        GitTaskId(
            git_url=_GIT_URL,
            git_commit_id=_GIT_COMMIT,
            path=Path(_DATASET_PREFIX) / name,
        )
        for name in task_names
    ]

    client = TaskClient()
    logger.info("Downloading %d tasks from %s …", len(task_ids), _GIT_URL)
    paths = client.download_tasks(task_ids, overwrite=overwrite, output_dir=output_dir)
    logger.info("Downloaded %d tasks", len(paths))
    return paths


# =============================================================================
# Task directory scanner (for pre-downloaded local dirs)
# =============================================================================


def _is_valid_task(entry: Path) -> bool:
    if not entry.is_dir():
        return False
    return (entry / "task.toml").exists() or any(
        (entry / f).exists() for f in _INSTRUCTION_FILES
    )


def scan_task_dirs(
    root: Path,
    task_names: list[str] | None = None,
) -> list[Path]:
    """Scan *root* for Harbor task directories (also checks one level deep).

    Args:
        root:       Directory containing task subdirectories (may be nested
                    one level, e.g. harbor's ``~/.cache/harbor/tasks/``).
        task_names: Optional allowlist of directory names.

    Returns:
        Sorted list of valid task directory paths.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Task root not found: {root}")

    tasks: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if _is_valid_task(entry):
            if not task_names or entry.name in task_names:
                tasks.append(entry)
        else:
            # One level deeper — harbor cache layout: {uuid}/{task_name}/
            for child in sorted(entry.iterdir()):
                if not child.is_dir():
                    continue
                if _is_valid_task(child):
                    if not task_names or child.name in task_names:
                        tasks.append(child)

    return tasks


# =============================================================================
# Parquet builder
# =============================================================================


def build_parquet(task_dirs: list[Path], output_path: Path) -> Path:
    """Write a SLIME harbor parquet from a list of task directories.

    Args:
        task_dirs:   Task directory paths (each dir name is the task_id).
        output_path: Destination parquet file (parent dirs created).

    Returns:
        The *output_path* written.
    """
    if not task_dirs:
        raise ValueError("No task directories provided")

    rows = []
    for task_dir in task_dirs:
        task_id = task_dir.name
        rows.append({
            "task_id": task_id,
            "metadata": {
                "instance": {
                    "task_id": task_id,
                    "task_dir": str(task_dir.resolve()),
                },
            },
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return output_path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build a SLIME harbor parquet (task_id only — no prompt text)"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download tasks from the harbor-datasets git repo via TaskClient "
            "(cached under ~/.cache/harbor/tasks/). Without --tasks, downloads all tasks."
        ),
    )
    source.add_argument(
        "--task-dir",
        type=Path,
        metavar="DIR",
        help="Local directory already containing Harbor task subdirectories.",
    )

    parser.add_argument(
        "--output", type=Path,
        default=Path("/root/harbor-data/swebench_verified/train.parquet"),
        help="Output parquet path (default: /root/harbor-data/swebench_verified/train.parquet)",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Specific task IDs to include. Required with --download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download tasks even if already cached (only with --download).",
    )
    args = parser.parse_args()

    if args.download:
        task_dirs = download_tasks(
            task_names=args.tasks or None,
            overwrite=args.overwrite,
        )
    else:
        task_dirs = scan_task_dirs(args.task_dir, task_names=args.tasks)
        if not task_dirs:
            logger.error("No valid task directories found under %s", args.task_dir)
            return

    print(f"Found {len(task_dirs)} tasks")
    build_parquet(task_dirs, args.output)
    print(f"Wrote {len(task_dirs)} rows to {args.output}")


if __name__ == "__main__":
    main()
