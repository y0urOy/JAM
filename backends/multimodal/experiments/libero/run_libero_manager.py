import os
import math
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _resolve_tagged_output_dir(raw_output_dir: Path, ckpt_tag: str) -> Path:
    run_ts = raw_output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run timestamp): {raw_output_dir}")

    eval_root = (PROJECT_ROOT / "evaluate_results").resolve()
    try:
        relative = raw_output_dir.resolve().relative_to(eval_root)
    except ValueError:
        if raw_output_dir.parent.name == ckpt_tag:
            return raw_output_dir.resolve()
        return (raw_output_dir.parent / ckpt_tag / run_ts).resolve()

    if len(relative.parts) >= 2 and relative.parts[1] == ckpt_tag:
        return raw_output_dir.resolve()

    family = relative.parts[0] if len(relative.parts) > 0 else "libero"
    return (eval_root / family / ckpt_tag / run_ts).resolve()


def _parse_gpu_ids(raw_gpu_ids: Any) -> list[int] | None:
    if raw_gpu_ids is None:
        return None
    if isinstance(raw_gpu_ids, str):
        text = raw_gpu_ids.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        gpu_ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        gpu_ids = [int(item) for item in raw_gpu_ids]

    if not gpu_ids:
        raise ValueError("MULTIRUN.gpu_ids must contain at least one GPU id when set.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"MULTIRUN.gpu_ids contains duplicates: {gpu_ids}")
    return gpu_ids


def _resolve_sample_ratio(raw_ratio: Any) -> float | None:
    if raw_ratio is None:
        return None
    if isinstance(raw_ratio, str):
        text = raw_ratio.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        ratio = float(text)
    else:
        ratio = float(raw_ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"MULTIRUN.task_sample_ratio must be in (0, 1], got {ratio}")
    if ratio >= 1.0:
        return None
    return ratio


def create_task_file(
    output_file: Path,
    task_suite_names: list[str],
    *,
    sample_ratio: float | None = None,
    sample_seed: int = 42,
) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            task_ids = list(range(n_tasks))
            if sample_ratio is not None:
                n_sample = max(1, int(math.ceil(n_tasks * sample_ratio)))
                rng = random.Random(f"{sample_seed}:{suite_name}")
                task_ids = sorted(rng.sample(task_ids, n_sample))
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            if sample_ratio is not None:
                print(
                    f"- Sampled tasks: {len(task_ids)} "
                    f"(ratio={sample_ratio:g}, seed={sample_seed})"
                )
            for task_id in task_ids:
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "model_overrides",
        "model_overrides_path",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.task_chunk_file",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def run_evaluation(
    *,
    task_file: Path,
    config_name: str,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    chunk_size: int,
    output_dir: Path,
    extra_overrides: list[str],
    gpu_ids: list[int] | None,
) -> None:
    script_path = PROJECT_ROOT / "experiments" / "libero" / "run_libero_parallel_test.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = str(PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    effective_num_gpus = len(gpu_ids) if gpu_ids is not None else num_gpus

    env = os.environ.copy()
    env.update(
        {
            "CONFIG_NAME": config_name,
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(effective_num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "TASK_CHUNK_SIZE": str(chunk_size),
            "ROOT_DIR": root_dir,
            "RUN_ID": output_dir.name,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
        }
    )
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    print("\nStarting evaluation (Hydra manager)...")
    print(f"config: {config_name}")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {effective_num_gpus}")
    if gpu_ids is not None:
        print(f"GPU ids: {gpu_ids}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Task chunk size: {chunk_size}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding='utf-8'))
        raise


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    config_name = HydraConfig.get().job.config_name or "sim_libero"
    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    raw_output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir = _resolve_tagged_output_dir(raw_output_dir, ckpt_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = _resolve_path(str(task_file_cfg), base=PROJECT_ROOT)
        if not task_file.exists():
            raise FileNotFoundError(f"MULTIRUN.task_file not found: {task_file}")
    else:
        task_file = output_dir / "tasks.txt"
        task_file = create_task_file(
            task_file,
            list(manager.task_suite_names),
            sample_ratio=_resolve_sample_ratio(manager.get("task_sample_ratio", None)),
            sample_seed=int(manager.get("task_sample_seed", 42)),
        )

    gpu_ids = _parse_gpu_ids(manager.get("gpu_ids", None))
    model_overrides_path = output_dir / "model_overrides.yaml"
    model_overrides = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    OmegaConf.save(config=model_overrides, f=str(model_overrides_path))
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    extra_overrides = collect_worker_overrides()
    extra_overrides.append(f"model_overrides_path={str(model_overrides_path)}")

    run_evaluation(
        task_file=task_file,
        config_name=str(config_name),
        task_choice=task_choice,
        ckpt=str(ckpt_path),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        chunk_size=max(1, int(manager.get("chunk_size", 1))),
        output_dir=output_dir,
        extra_overrides=extra_overrides,
        gpu_ids=gpu_ids,
    )


if __name__ == "__main__":
    main()
