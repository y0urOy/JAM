# Evaluation

Always use the exact checkpoint, matching normalization statistics, and matching robot/data recipe. Evaluation never runs the training-only future anchor and predictor.

Both backends use the same interface. New runs created by `tools/jam.py train`
remember their recipe and training overrides:

```bash
python tools/jam.py eval --run-dir artifacts/runs/wan-jam \
  --venv .venv --benchmark libero --gpus 0,1
python tools/jam.py eval --run-dir artifacts/runs/ovis-jam \
  --venv .venv --benchmark plus --gpus 0,1
python tools/jam.py eval --run-dir artifacts/runs/robotwin-qwen-jam \
  --venv .venv --benchmark robotwin --gpus 0,1
```

The latest nonempty numeric checkpoint is selected automatically. To select an
earlier step, add `--checkpoint` pointing inside that run's `checkpoints/weights/`.
For old runs without `run.json`, supply the recipe, checkpoint and statistics
explicitly, plus any original model/preprocessing overrides:

```bash
python tools/jam.py eval qwen-jam --benchmark libero \
  --checkpoint /path/to/step_021700.pt --stats /path/to/dataset_stats.json --gpus 0,1
python tools/jam.py eval qwen-jam --benchmark plus \
  --checkpoint /path/to/step_021700.pt --stats /path/to/dataset_stats.json --gpus 0,1
```

For the released Hugging Face checkpoints, use each folder's `release.py`, which applies their recorded settings.

FLUX.2 encodes each instruction during evaluation with the downloaded Qwen3-4B
(`qwen3` asset), padded to 512 tokens. DINOv2 instead reads its instruction features from the Qwen3-4B text cache (see
LIBERO-Plus below).

## LIBERO and LIBERO-Plus

The six LIBERO recipes (`wan-jam`, `ovis-jam`, `flux-jam-libero`, `qwen-jam`, `dino-b-jam`, `dino-l-jam`) are evaluated on both benchmarks.

LIBERO uses 40 tasks, 50 rollouts each, four suites. Report `Spatial | Object | Goal | Long | AVG`. LIBERO-Plus uses the fixed manifest of 1,506 perturbed tasks, seed 42, one trial per task, with a task-weighted micro average. Its manifest includes zero-based IDs; the original upstream classification uses one-based IDs. Do not compute an unweighted mean of perturbation-category rates.

The paper's LIBERO-Plus table splits the same results into seven perturbation categories:

```bash
python backends/wan/experiments/libero/summarize_libero_plus_results.py \
  --output-dir artifacts/evaluations/ovis-jam/plus/step_021700 --protocol jam_multimodal-paper15 \
  --classification-file third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json
```

This protocol joins categories as the released ImageWAM tables do (`classification[task_id - 1]`);
`--category-index-mode correct` uses `classification[task_id]`.

The portable runner uses one worker per listed GPU and scalar policy inference.
LIBERO/Plus share task scheduling, canonical result names and aggregation across
JAM backends. Wan and Qwen3.5 replan every 10 steps; Ovis-U1 and FLUX.2 every 8; DINOv2 every 8 on LIBERO and
every 10 on LIBERO-Plus, as in the recorded evaluations. For the released
checkpoints, each folder's `release.py` applies the recorded settings listed in `reproduction/libero_checkpoints.json`.

The DINOv2 text cache built for training holds the 40 LIBERO instructions. Before a
LIBERO-Plus evaluation of `dino-b-jam` or `dino-l-jam`, add the 1,502 distinct
instructions of the fixed task list to the same cache, on a CUDA GPU:

```bash
cd backends/multimodal
python experiments/libero/precompute_libero_plus_flux2_qwen3_cache.py \
  --libero-plus-root ../../third_party/LIBERO-plus \
  --cache-dir ../../artifacts/cache/libero/qwen3_4b_len512 \
  --model-spec ../../artifacts/models/Qwen3-4B
```

`--validate-only` checks an existing cache without loading Qwen3-4B. The script
refuses a LIBERO-Plus checkout other than the pinned commit.

Results go to `<run>/evaluations/<benchmark>/<checkpoint>/` with `--run-dir`, or
`artifacts/evaluations/<recipe>/<benchmark>/<checkpoint>/` otherwise. Override
with `--output`; it must be empty so old tasks cannot contaminate a new score. `--resume` finishes an interrupted
evaluation in the same output (same checkpoint, statistics and overrides): valid task results are kept and only
the missing tasks run. Within a run, a worker whose simulator crashes, or finishes no task for two hours, is
stopped, and its unfinished tasks run again in fresh workers (up to two more attempts).

| File | Contents |
|---|---|
| `evaluation_plan.json` | Checkpoint, statistics, overrides, assigned tasks and commands |
| `worker_*.log`, `tasks_gpu*.txt` | Per-GPU logs and task lists |
| `execution_report.json` | Process status and result completeness |
| `summary.json`, `summary.csv` | Validated success rates; Spatial, Object, Goal, Long, AVG |
| `episodes.csv` | `task_id,episode,success` for every episode; empty if incomplete |

Aggregation requires every expected task exactly once, the correct trial count,
and disjoint success/failure episode IDs covering every trial. Missing,
unexpected, duplicate or malformed results produce `complete=false`, null
aggregate scores and a nonzero exit. If retries leave tasks unfinished, no complete score is reported. Inference overrides are recorded with the results.

Recheck an existing result directory without running any model:

```bash
python tools/jam.py summarize --benchmark libero \
  --output artifacts/runs/ovis-jam/evaluations/libero/step_021700
```

## RoboTwin 2.0

The four RoboTwin recipes evaluate all 50 tasks under Clean and Randomized settings, 100 episodes per task/setting, unseen instructions and the official expert-seed validity filter:

```bash
python tools/jam.py eval robotwin-wan-jam --venv .venv --benchmark robotwin \
  --checkpoint /path/to/step_029355.pt --stats /path/to/dataset_stats.json --gpus 0,1
python tools/jam.py eval robotwin-ovis-jam --venv .venv --benchmark robotwin \
  --checkpoint /path/to/step_069900.pt --stats /path/to/dataset_stats.json --gpus 0,1
python tools/jam.py eval robotwin-flux-jam --venv .venv --benchmark robotwin \
  --checkpoint /path/to/step_069900.pt --stats /path/to/dataset_stats.json --gpus 0,1
python tools/jam.py eval robotwin-qwen-jam --venv .venv --benchmark robotwin \
  --checkpoint /path/to/step_069900.pt --stats /path/to/dataset_stats.json --gpus 0,1
```

The launcher runs `experiments/robotwin/run_robotwin_manager.py --config-name sim_robotwin`
in the recipe's backend, one task at a time per listed GPU. It writes RoboTwin logs
and its native `summary.json` / `summary.csv` (per-task Clean and Randomized success
rates; `complete`, which is true when every task finished in every setting; and, for
a complete evaluation, the Clean and Randomized means and their average) directly to
the selected output directory. It loads the task
list from the configured simulator root. The backend links its policy adapter as
`third_party/RoboTwin/policy/jam_policy`; the link points to one backend at a time,
so remove it before switching between a Wan recipe and a multimodal recipe (the
evaluator stops with a symlink conflict otherwise). Install the simulator packages and
assets as described in [INSTALL.md](INSTALL.md); the evaluator requires SAPIEN 3.0.0b1. RoboTwin retains its
Clean/Randomized result schema; the LIBERO `summarize` command is not a RoboTwin
result validator.
