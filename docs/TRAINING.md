# Training

JAM variants share `tools/jam.py`. `list --backend wan` or
`list --backend multimodal` filters the recipes. The recipe JSON supplies scientific
settings; the launcher handles paths, topology, logs and run metadata. Planning
with `--dry-run` uses only the standard library and does not create any files.

Use `--venv .venv-wan`, `--venv .venv-ovis`, etc. to choose the runtime without
activating it; `--python /path/to/python` also works. Without either option, the
current Python is used. Keep the environment profiles separate.

## Train and evaluate

```bash
python tools/jam.py train-eval wan-jam --venv .venv-wan \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/wan-jam
python tools/jam.py train-eval ovis-jam --venv .venv-ovis \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/ovis-jam
```

Use `wan-generation` / `ovis-generation` for the FastWAM / ImageWAM baselines. Each command trains **one** model, then evaluates
that model's final checkpoint. It stops on any stage failure. With no
`--benchmarks`, it evaluates the recipe's training benchmark (LIBERO or RoboTwin).
Install the requested simulators and assets before launching a long training run.
`--gpus-per-node` controls training; `--gpus` selects local evaluation GPUs.

Training overrides are also applied during evaluation so architecture and
preprocessing choices stay paired. Use `--eval-override` for evaluation-only
changes, for example `--eval-override EVALUATION.num_inference_steps=10`.
The launcher owns output paths, task selection, statistics and trial counts;
use backend scripts for custom protocols outside these fixed benchmarks.

## Train only

The existing `train` command remains available. Activate the matching environment
or supply `--venv` in these examples:

```bash
python tools/jam.py train ovis-jam --gpus-per-node 8
python tools/jam.py train qwen-jam --gpus-per-node 8
python tools/jam.py train dino-l-jam --gpus-per-node 8
python tools/jam.py train robotwin-wan-jam --gpus-per-node 8 --micro-batch 32
```

On eight separate one-GPU hosts, run the same command on every host with `--nodes 8 --gpus-per-node 1 --node-rank RANK --master-addr ADDRESS`. The launcher does not submit scheduler jobs. With fewer GPUs, accumulation retains the global batch; memory limits may require a smaller `--micro-batch`. A changed topology does not guarantee bitwise-identical floating-point training.

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

The native 24-layer Qwen action expert is the paper architecture; the earlier Wan-initialized Qwen adapter remains reference source, not the `qwen-jam` recipe. DINO uses depth/width-matched native action blocks and frozen Qwen3-4B text, not the older Yak action interface.

Readout offsets have different meanings by family. Wan uses causal-VAE positions `[1,2]`; the other main variants use the recorded 16-step endpoint. Qwen LIBERO's logical `[4]` corresponds to `t+16` under the dataset grid. RoboTwin Qwen uses logical `[16]` with endpoint-only decoding. Preserve these mappings.

The Wan 10k budget controls have their own full cosine schedule. They are not intermediate checkpoints from a 20k schedule. EMA experiments should be reproduced from scratch; a weights-only resume cannot reconstruct EMA history.
