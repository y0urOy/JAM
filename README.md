# JAM

Official codebase for **Vision-Language-Action or World-Action? Just Action Models**.

[![Project Page](https://img.shields.io/badge/Project_Page-JAM-2ea44f.svg)](https://y0uroy.github.io/JAM/)
![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/y0urOy/JAM)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

This repository contains the training and evaluation code for JAM on LIBERO, LIBERO-Plus and RoboTwin 2.0.
JAM trains a pretrained backbone and an action expert with a foresight loss: a small head predicts frozen
features of the recorded future from the current frame. No image or video is generated in training or at
deployment. The code covers five backbones: Wan2.2, built on [FastWAM](https://github.com/yuantianyuan01/FastWAM),
and Ovis-U1, FLUX.2, Qwen3.5-2B and DINOv2-B/L, built on [ImageWAM](https://github.com/yuyangalin/ImageWAM).

<p align="center">
  <img src="website/assets/figures/fig2_method.png" alt="JAM: future-feature supervision during training; backbone and action expert at deployment" width="100%">
</p>

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [BibTeX](#bibtex)

## File Structure

```text
JAM/
├── backends/
│   ├── wan/                  # JAM (Wan), built on FastWAM
│   └── multimodal/           # JAM (Ovis-U1, FLUX.2, Qwen3.5, DINOv2), built on ImageWAM
├── tools/
│   ├── jam.py                # Train, evaluate and summarize any named recipe
│   ├── bootstrap.py          # Create the pinned environment of a backbone family
│   ├── download_assets.py    # Base models and datasets at pinned revisions
│   ├── init_actions.py       # Build the action-expert initialization
│   ├── setup_simulators.py   # LIBERO and LIBERO-Plus checkouts
│   └── paired_statistics.py  # Task-level bootstrap and McNemar tests
├── environments/             # Pinned dependencies: wan, ovis, flux, qwen
├── reproduction/             # 48 named recipes, LIBERO-Plus task list, checkpoint manifest
├── third_party/RoboTwin/     # RoboTwin 2.0 evaluation code
├── docs/                     # Installation, data, training, evaluation, analyses
├── paper/                    # The paper's tables and figure definitions
├── website/                  # Project page
└── tests/                    # CPU tests
```

## Environment Setup

Each backbone family has its own pinned environment, because the backbones need different Transformers
versions. Use Python 3.10:

| Policy | Recipe | Environment |
|---|---|---|
| JAM (Wan) | `wan-jam` | `wan` |
| JAM (Ovis-U1) | `ovis-jam` | `ovis` |
| JAM (DINOv2-B / L) | `dino-b-jam` / `dino-l-jam` | `ovis` |
| JAM (Qwen3.5-2B) | `qwen-jam` | `qwen` |
| JAM (FLUX.2) | `flux-jam-libero` | `flux` |

```bash
python3.10 tools/bootstrap.py wan --venv .venv-wan --libero
python3.10 tools/bootstrap.py ovis --venv .venv-ovis --libero
```

`--libero` also installs the LIBERO runtime. See [docs/INSTALL.md](docs/INSTALL.md) for the versions and the
optional Qwen kernels.

## Model Preparation

Download the base models at their pinned revisions. They go to `artifacts/models/` unless `JAM_MODEL_ROOT` is set.

| Recipes | Assets |
|---|---|
| Wan | `wan`, `wan-tokenizer` |
| Ovis-U1 | `ovis` |
| Qwen3.5-2B | `qwen-2b` (`qwen-4b` for the 4B ablation) |
| DINOv2-B / L | `dino-b` / `dino-l`, `qwen3` |
| FLUX.2 | `flux`, `flux-ae`, `qwen3` |

```bash
python tools/download_assets.py wan
python tools/download_assets.py wan-tokenizer
```

Training also needs the action-expert initialization (DINOv2 and Qwen3.5 initialize from their own backbone):

```bash
python tools/init_actions.py wan
python tools/init_actions.py ovis --action-dim 7    # 14 for RoboTwin
python tools/init_actions.py flux --action-dim 7    # 14 for RoboTwin
```

## Dataset Download

JAM uses the LIBERO and RoboTwin 2.0 datasets preprocessed by FastWAM:

```bash
python tools/download_assets.py libero-data
python tools/download_assets.py robotwin-data
```

Extract the archives in place (`tar -xzf` each LIBERO archive; `cat robotwin2.0.tar.gz.part-* | tar -xzf -` for
RoboTwin). The expected layout is:

```text
artifacts/data/
├── libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot/
└── robotwin2.0/
    ├── robotwin2.0/{data,meta,videos}/
    └── dataset_stats.json
```

Fetch the simulators with `python tools/setup_simulators.py libero` and `python tools/setup_simulators.py plus`.
Their assets and initial states come from the upstream projects. `third_party/RoboTwin/` holds the RoboTwin 2.0
evaluation code; install its environment and assets from an upstream checkout, as described in
[docs/INSTALL.md](docs/INSTALL.md).

## Inference with Released Checkpoints

The three LIBERO checkpoints are on [Hugging Face](https://huggingface.co/y0urOy/JAM), one folder per policy.
Their recorded results:

| Policy | Folder | Spatial | Object | Goal | Long | LIBERO | LIBERO-Plus |
|---|---|---:|---:|---:|---:|---:|---:|
| JAM (Wan) | [`libero/wan`](https://huggingface.co/y0urOy/JAM/tree/main/libero/wan) | 98.80 | 100.00 | 98.00 | 95.80 | 98.15 | 64.54 |
| JAM (Ovis-U1) | [`libero/ovis-u1`](https://huggingface.co/y0urOy/JAM/tree/main/libero/ovis-u1) | 98.60 | 99.60 | 98.40 | 90.40 | 96.75 | 79.95 |
| JAM (Qwen3.5-2B) | [`libero/qwen3.5-2b`](https://huggingface.co/y0urOy/JAM/tree/main/libero/qwen3.5-2b) | 97.80 | 99.60 | 97.20 | 91.60 | 96.55 | 81.14 |

LIBERO: 40 tasks × 50 episodes. LIBERO-Plus: the fixed 1,506-task subset, one episode per task, task-weighted.

```bash
pip install -U huggingface_hub
huggingface-cli download y0urOy/JAM --include "libero/wan/*" --local-dir ./checkpoints/jam_release
python checkpoints/jam_release/libero/wan/release.py verify
```

Each folder's `README.md` gives the commands that evaluate it with its recorded settings through `release.py eval`,
from a checkout of the JAM commit that the folder pins. The [checkpoint manifest](reproduction/libero_checkpoints.json)
lists their SHA256 checksums and inference settings.

## Training

`tools/jam.py` runs the 48 named recipes in [recipes.json](reproduction/recipes.json): JAM policies, the FastWAM and
ImageWAM generation baselines, action-only controls and ablations.

```bash
python tools/jam.py list
python tools/jam.py text-cache wan-jam    # instruction cache (Wan, FLUX.2 and DINOv2 recipes)
python tools/jam.py train-eval wan-jam --venv .venv-wan \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/wan-jam
```

`train-eval` trains one model, then evaluates its final checkpoint, and stops on any failure. Use
`wan-generation` or `ovis-generation` for the FastWAM or ImageWAM baselines. The launcher keeps the recipe's
global batch on any GPU layout (`--micro-batch` fits memory, `--nodes`/`--node-rank`/`--master-addr` run
multi-node), and `--dry-run` prints the plan without running it. See [docs/TRAINING.md](docs/TRAINING.md) for
training budgets and the precomputed foresight-target cache.

## Inference with Your Trained Checkpoints

A run directory remembers its recipe, overrides and normalization statistics:

```bash
python tools/jam.py eval --run-dir artifacts/runs/wan-jam --venv .venv-wan --benchmark libero --gpus 0,1
python tools/jam.py eval --run-dir artifacts/runs/wan-jam --venv .venv-wan --benchmark plus --gpus 0,1
python tools/jam.py eval --run-dir artifacts/runs/robotwin-wan-jam --venv .venv-wan --benchmark robotwin --gpus 0
```

LIBERO and LIBERO-Plus results are written as `summary.json` and `summary.csv`, with per-episode `episodes.csv`
for `tools/paired_statistics.py`. RoboTwin 2.0 uses 50 tasks × 100 episodes in Clean and Randomized settings. See
[docs/EVALUATION.md](docs/EVALUATION.md) for the protocols and [docs/ANALYSES.md](docs/ANALYSES.md) for the
future-frame probes, attention maps and compute profiling.

## Acknowledgements

JAM builds on [FastWAM](https://github.com/yuantianyuan01/FastWAM) and [ImageWAM](https://github.com/yuyangalin/ImageWAM).
The RoboTwin evaluation code is adapted from [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and the
data readers from [LeRobot](https://github.com/huggingface/lerobot). We thank these teams, and the authors of
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus),
for releasing their code, data and assets.

## License

JAM additions are released under the [MIT License](LICENSE). Code from FastWAM, ImageWAM, RoboTwin, LeRobot,
Ovis-U1 and FLUX.2 keeps its own license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights,
datasets and simulator assets keep their own terms.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@misc{yu2026jam,
  title  = {Vision-Language-Action or World-Action? Just Action Models},
  author = {Yu, Yonghao and Huang, Lang and Li, Runyi and Yamasaki, Toshihiko},
  year   = {2026},
  url    = {https://y0uroy.github.io/JAM/}
}
```
