# Training

The ten JAM recipes share `tools/jam.py`. `list --backend wan` or
`list --backend multimodal` filters the recipes. The recipe JSON supplies scientific
settings; the launcher handles paths, topology, logs and run metadata. Planning
with `--dry-run` uses only the standard library and does not create any files.

| Policy | LIBERO recipe | RoboTwin 2.0 recipe |
|---|---|---|
| JAM (Wan) | `wan-jam` | `robotwin-wan-jam` |
| JAM (Ovis-U1) | `ovis-jam` | `robotwin-ovis-jam` |
| JAM (FLUX.2) | `flux-jam-libero` | `robotwin-flux-jam` |
| JAM (Qwen3.5-2B) | `qwen-jam` | `robotwin-qwen-jam` |
| JAM (DINOv2-B / L) | `dino-b-jam` / `dino-l-jam` | |

All ten recipes run in the environment that `tools/bootstrap.py` installs. Use
`--venv .venv` to choose it without activating it; `--python /path/to/python` also
works. Without either option, the current Python is used.

Before training, download the assets and build the action initialization and the
instruction cache that the recipe needs; see [DATA_AND_MODELS.md](DATA_AND_MODELS.md).

## Train and evaluate

```bash
python tools/jam.py train-eval wan-jam --venv .venv \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/wan-jam
python tools/jam.py train-eval ovis-jam --venv .venv \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/ovis-jam
python tools/jam.py train-eval robotwin-ovis-jam --venv .venv \
  --gpus-per-node 8 --gpus 0,1 --output artifacts/runs/robotwin-ovis-jam
```

Each command trains **one** model, then evaluates that model's final checkpoint.
It stops on any stage failure. With no `--benchmarks`, it evaluates the recipe's
training benchmark: LIBERO or RoboTwin. LIBERO recipes also accept `plus`.
Install the requested simulators and assets before launching a long training run.
`--gpus-per-node` controls training; `--gpus` selects local evaluation GPUs.

Training overrides are also applied during evaluation so architecture and
preprocessing choices stay paired. Use `--eval-override` for evaluation-only
changes, for example `--eval-override EVALUATION.num_inference_steps=10`.
The launcher owns output paths, task selection, statistics and trial counts;
use backend scripts for custom protocols outside these fixed benchmarks.

## Train only

The `train` command trains without evaluating. Activate the environment or
supply `--venv` in these examples:

```bash
# LIBERO
python tools/jam.py train wan-jam --gpus-per-node 8
python tools/jam.py train ovis-jam --gpus-per-node 8
python tools/jam.py train flux-jam-libero --gpus-per-node 8
python tools/jam.py train qwen-jam --gpus-per-node 8
python tools/jam.py train dino-b-jam --gpus-per-node 8
python tools/jam.py train dino-l-jam --gpus-per-node 8
# RoboTwin 2.0
python tools/jam.py train robotwin-wan-jam --gpus-per-node 8 --micro-batch 32
python tools/jam.py train robotwin-ovis-jam --gpus-per-node 8
python tools/jam.py train robotwin-flux-jam --gpus-per-node 8
python tools/jam.py train robotwin-qwen-jam --gpus-per-node 8
```

On eight separate one-GPU hosts, run the same command on every host with `--nodes 8 --gpus-per-node 1 --node-rank RANK --master-addr ADDRESS`. The launcher does not submit scheduler jobs. With fewer GPUs, accumulation retains the global batch; memory limits may require a smaller `--micro-batch`. A changed topology does not guarantee bitwise-identical floating-point training.

On the pinned PyTorch 2.7 CUDA runtime, very large flattened AdamW partitions can exceed the foreach optimizer's safe indexing range. For Wan or FLUX.2 training with fewer GPUs, add `--override optimizer_max_group_numel=1000000000` to split equal-hyperparameter groups before DeepSpeed flattens them. Reducing the micro-batch does not reduce optimizer-state sizes. The eight-GPU recipes keep their recorded parameter grouping.

Multi-node runs require a shared output directory. With `train-eval`, only node
rank 0 starts evaluation, using that node's `--gpus`; the other nodes exit after
training. `--workers` controls DataLoader workers. Every new run requires a fresh
`--output`, protecting previous weights and logs. For full optimizer-state
resumption, use the original backend training script with its `resume` setting;
the unified launcher does not infer resumption from an existing directory.

## Run artifacts

```text
artifacts/runs/ovis-jam/
  run.json                    # recipe snapshot, overrides, source version, paths, status
  train_node0.log             # one log per training node
  config.yaml                 # resolved backend configuration
  dataset_stats.json          # this run's normalization statistics
  checkpoints/weights/step_021700.pt
  evaluations/libero/step_021700/
  evaluations/plus/step_021700/
```

`run.json` records SHA256 checksums for the small configuration/statistics files
after training. `eval --run-dir` verifies these, restores training overrides and
selects the highest nonempty numeric `step_*.pt`. It refuses a different recipe
or statistics file. Move the entire run directory together; use `--model-root`,
`--data-root` and `--cache-root` to relocate dependencies. Absolute paths supplied
inside custom `--override` values must be updated explicitly.

## Scientific settings

| Recipe group | Updates | Global batch | Action horizon |
|---|---:|---:|---:|
| LIBERO Wan | 20,000 | 128 | 32 |
| LIBERO Ovis-U1/Qwen3.5/DINOv2/FLUX.2 | 21,700 | 128 | 16 |
| RoboTwin Wan | 29,355 | 1,024 | 32 |
| RoboTwin Ovis-U1/Qwen3.5/FLUX.2 | 69,900 | 384 | 16 |

Training uses DeepSpeed with BF16 at ZeRO stage 1, or stage 0 for `robotwin-ovis-jam` and the DINOv2
recipes (the recipe's `zero_stage`; see `environments/ds_zero*.json`). These settings apply no gradient
clipping, so `max_grad_norm` in the backend configs has no effect.

The 69,900 RoboTwin Ovis-U1, Qwen3.5 and FLUX.2 updates are five epochs of the 5,368,277 non-idle training windows
(`nonidle_ranges.json`; see [DATA_AND_MODELS.md](DATA_AND_MODELS.md)); the RoboTwin Wan recipe trains on all frames.
The launcher stops a multimodal run at the first non-finite loss (`JAM_FAIL_ON_NONFINITE_LOSS=1`).

The Qwen3.5 recipes use the native 24-layer Qwen action expert. DINOv2 uses depth/width-matched native action blocks and frozen Qwen3-4B instruction features read from the text cache.

Readout offsets have different meanings by family. Wan uses causal-VAE positions `[1,2]`; the other variants use the recorded 16-step endpoint. Qwen LIBERO's logical `[4]` corresponds to `t+16` under the dataset grid. RoboTwin Qwen uses logical `[16]` with endpoint-only decoding. Preserve these mappings.
