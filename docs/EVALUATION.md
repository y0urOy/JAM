# Evaluation

Always use the exact checkpoint, matching normalization statistics, and matching robot/data recipe. Evaluation drops the training-only future anchor and predictor.

Both backends use the same interface. New runs created by `tools/jam.py train`
remember their recipe and training overrides:

```bash
python tools/jam.py eval --run-dir artifacts/runs/wan-jam \
  --venv .venv-wan --benchmark libero --gpus 0,1
python tools/jam.py eval --run-dir artifacts/runs/ovis-jam \
  --venv .venv-ovis --benchmark plus --gpus 0,1
```

The latest nonempty numeric checkpoint is selected automatically. To select an
earlier step, add `--checkpoint` pointing inside that run's `checkpoints/weights/`.
For old runs without `run.json`, supply the recipe, checkpoint and statistics
explicitly, plus any original model/preprocessing overrides. For the released
Hugging Face checkpoints, use each folder's `release.py`, which applies their
recorded settings:

```bash
python tools/jam.py eval qwen-jam --benchmark libero \
  --checkpoint /path/to/step_021700.pt --stats /path/to/dataset_stats.json --gpus 0,1
python tools/jam.py eval qwen-jam --benchmark plus \
  --checkpoint /path/to/step_021700.pt --stats /path/to/dataset_stats.json --gpus 0,1
```

LIBERO uses 40 tasks, 50 rollouts each, four suites. Report `Spatial | Object | Goal | Long | AVG`. LIBERO-Plus uses the fixed manifest of 1,506 perturbed tasks, seed 42, one trial per task, with a task-weighted micro average. Its manifest includes zero-based IDs; the original upstream classification uses one-based IDs. Do not compute an unweighted mean of perturbation-category rates.

The portable runner uses one worker per listed GPU and scalar policy inference.
LIBERO/Plus share task scheduling, canonical result names and aggregation across
JAM backends. Wan retains replan 10; Ovis-U1 retains replan 8.

Results go to `<run>/evaluations/<benchmark>/<checkpoint>/` with `--run-dir`, or
`artifacts/evaluations/<recipe>/<benchmark>/<checkpoint>/` otherwise. Override
with `--output`; it must be empty so old tasks cannot contaminate a new score.

| File | Contents |
|---|---|
| `evaluation_plan.json` | Checkpoint, statistics, overrides, assigned tasks and commands |
| `worker_*.log`, `tasks_gpu*.txt` | Per-GPU logs and task lists |
| `execution_report.json` | Process status and result completeness |
| `summary.json`, `summary.csv` | Validated success rates; Spatial, Object, Goal, Long, AVG |
| `episodes.csv` | `task_id,episode,success` for paired comparisons; empty if incomplete |

Aggregation requires every expected task exactly once, the correct trial count,
and disjoint success/failure episode IDs covering every trial. Missing,
unexpected, duplicate or malformed results produce `complete=false`, null
aggregate scores and a nonzero exit. A worker failure also prevents a complete
score. Inference overrides are recorded; using them does not establish parity
with the paper protocol.

Recheck an existing result directory without running any model:

```bash
python tools/jam.py summarize --benchmark libero \
  --output artifacts/runs/ovis-jam/evaluations/libero/step_021700
```

RoboTwin evaluates all 50 tasks under Clean and Randomized settings, 100 episodes per task/setting, unseen instructions and the official expert-seed validity filter:

```bash
python tools/jam.py eval robotwin-wan-jam --benchmark robotwin \
  --checkpoint /path/to/step_029355.pt --stats /path/to/dataset_stats.json --gpus 0
```

The unified launcher writes RoboTwin logs and its native `summary.json` /
`summary.csv` directly to the selected output directory. It loads the task list
from the configured simulator root; the backend creates its policy symlink.
Configure simulator assets and controller dependencies using the included
RoboTwin README. RoboTwin retains its Clean/Randomized result schema; the
LIBERO `summarize` command is not a RoboTwin result validator.

For paired inference statistics, export CSVs with `task_id,episode,success` (success 0 or 1), preserving the same episode identities. Qualify task IDs by suite, for example `libero_spatial:0`, so task 0 from different suites is never merged:

```bash
python tools/paired_statistics.py --a baseline.csv --b jam.csv --out paired.json
```

The script rejects mismatched or duplicate pairs, uses 100,000 task-cluster bootstrap samples by default and an exact two-sided McNemar test. Bootstrap intervals quantify fixed-checkpoint evaluation uncertainty, not across-training-seed variance.
