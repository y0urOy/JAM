<div align="center">
  <h1>Just Action Models</h1>
  <h3>Vision-Language-Action or World-Action?</h3>
  <p>
    Yonghao Yu<sup>1</sup> &nbsp;·&nbsp;
    Lang Huang<sup>2,✉</sup> &nbsp;·&nbsp;
    Runyi Li<sup>1</sup> &nbsp;·&nbsp;
    Toshihiko Yamasaki<sup>1</sup>
  </p>
  <p><sup>1</sup>The University of Tokyo &nbsp;&nbsp; <sup>2</sup>National Institute of Informatics</p>
  <p>
    <a href="https://y0uroy.github.io/JAM/"><img src="docs/assets/badges/website.svg" alt="Project page" height="20"></a>
    <img src="docs/assets/badges/arxiv.svg" alt="arXiv: coming soon" title="arXiv identifier pending" height="20">
    <a href="https://huggingface.co/y0urOy/JAM"><img src="docs/assets/badges/models.svg" alt="Hugging Face models" height="20"></a>
    <a href="docs/INSTALL.md"><img src="docs/assets/badges/docs.svg" alt="Documentation" height="20"></a>
  </p>
</div>

<p align="center">
  <b>Learn from the future. Act in the present.</b><br>
  Future observations supervise training. No image or video generation is needed.
</p>

<p align="center">
  <img src="website/assets/figures/fig2_method.png" alt="JAM method: future-feature supervision during training; backbone and action expert at deployment" width="100%">
</p>

JAM supports Wan, Ovis-U1, FLUX.2, Qwen3.5 and DINOv2-B/L, with shared training
and evaluation tools.

## Start

Select a JAM variant and its environment. All variants share the same training
and evaluation commands; each environment retains its pinned dependencies.

| Policy | Recipe | Environment |
|---|---|---|
| JAM (Wan) | `wan-jam` | `wan` |
| JAM (Ovis-U1) | `ovis-jam` | `ovis` |
| JAM (Qwen3.5-2B) | `qwen-jam` | `qwen` |
| JAM (DINOv2-B / L) | `dino-b-jam` / `dino-l-jam` | `ovis` |
| JAM (FLUX.2) | `flux-jam-libero` | `flux` |

Use Python 3.10 to prepare each environment:

```bash
python3.10 tools/bootstrap.py wan --venv .venv-wan --libero
python3.10 tools/bootstrap.py ovis --venv .venv-ovis --libero
python tools/jam.py list
python tools/jam.py train-eval wan-jam --venv .venv-wan --dry-run
```

Prepare the [datasets, models, initialization and text cache](docs/DATA_AND_MODELS.md)
and [simulators](docs/INSTALL.md). Train and evaluate with one command:

```bash
python tools/jam.py train-eval wan-jam --venv .venv-wan \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/wan-jam
python tools/jam.py train-eval ovis-jam --venv .venv-ovis \
  --gpus-per-node 8 --gpus 0,1 --benchmarks libero plus \
  --output artifacts/runs/ovis-jam

# Or train now and evaluate later; the run remembers its recipe and statistics.
python tools/jam.py train ovis-jam --venv .venv-ovis \
  --gpus-per-node 8 --output artifacts/runs/ovis-jam
python tools/jam.py eval --run-dir artifacts/runs/ovis-jam \
  --venv .venv-ovis --benchmark libero --gpus 0,1
```

Use `wan-generation` for the FastWAM baseline or `ovis-generation` for ImageWAM.
Implementation names and legacy compatibility are documented in [JAM naming](docs/NAMING.md).
`train-eval` stops on failure and requires the final checkpoint. Each benchmark
gets a separate result directory; LIBERO/Plus produce validated `summary.json`,
`summary.csv` and paired `episodes.csv`. Incomplete results have no aggregate score.

The launcher preserves the recipe's global batch when GPU topology changes. Use
`--micro-batch` to fit GPU memory. `--dry-run` plans without loading models or
creating runs. Paths can be relocated with `JAM_DATA_ROOT`, `JAM_MODEL_ROOT`,
`JAM_CACHE_ROOT`, and `JAM_ARTIFACT_ROOT`. See [training](docs/TRAINING.md) and
[evaluation](docs/EVALUATION.md) for multi-node and existing-checkpoint usage.

## Included

| Area | Entry |
|---|---|
| 48 recipes: JAM, baselines, controls and ablations | [recipes.json](reproduction/recipes.json), [training guide](docs/TRAINING.md) |
| LIBERO and fixed 1,506-task LIBERO-Plus | [evaluation guide](docs/EVALUATION.md), [task manifest](reproduction/libero_plus_paper15_tasks.txt) |
| RoboTwin 2.0 policy and simulator code | [evaluation guide](docs/EVALUATION.md), `third_party/RoboTwin/` |
| Future-frame probes, attention, FLOPs and GPU timing | [analysis guide](docs/ANALYSES.md) |
| Task bootstrap and paired tests | [paired_statistics.py](tools/paired_statistics.py) |
| Dependency pins, file checksums and checks | [environments](environments/), [FILE_MANIFEST.json](FILE_MANIFEST.json), [validation](VALIDATION.md) |

## Checkpoints

The three LIBERO paper checkpoints are released in one Hugging Face repository, [y0urOy/JAM](https://huggingface.co/y0urOy/JAM), with one folder per backbone. Each folder includes its matching configuration, normalization statistics, recorded LIBERO/Plus results, pinned dependencies and download/evaluation instructions.

| Model | Hugging Face folder |
|---|---|
| JAM (Wan) | [libero/wan](https://huggingface.co/y0urOy/JAM/tree/main/libero/wan) |
| JAM (Ovis-U1) | [libero/ovis-u1](https://huggingface.co/y0urOy/JAM/tree/main/libero/ovis-u1) |
| JAM (Qwen3.5-2B) | [libero/qwen3.5-2b](https://huggingface.co/y0urOy/JAM/tree/main/libero/qwen3.5-2b) |

Use each folder's `release.py` to verify and evaluate a policy with its original settings. The [checkpoint manifest](reproduction/libero_checkpoints.json) records SHA256 checksums and inference settings. Other paper models, such as the RoboTwin policies and the ablations, have no public weights; see [artifact status](reproduction/artifact_status.json).

## License

JAM additions are released under [MIT](LICENSE). FastWAM, ImageWAM, RoboTwin, LeRobot, Ovis-U1 and FLUX.2 notices are preserved; see [third-party notices](THIRD_PARTY_NOTICES.md). External datasets, model weights and simulator assets retain their own terms.
