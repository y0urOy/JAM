# Data and models

Download references and revisions are in [assets.json](../reproduction/assets.json). For `ovis`, `qwen-4b` and `qwen3`, the pins are upstream revisions and may differ from the ones used for the original training (see their `pin_evidence`). The trained LIBERO policies are on [Hugging Face](https://huggingface.co/y0urOy/JAM); see [libero_checkpoints.json](../reproduction/libero_checkpoints.json).

```bash
python tools/download_assets.py wan --dry-run
python tools/download_assets.py wan
python tools/download_assets.py wan-tokenizer
python tools/download_assets.py libero-data
```

The default layout is:

```text
artifacts/
  models/Wan-AI/Wan2.2-TI2V-5B/
  models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
  models/Ovis-U1-3B/
  models/Qwen3.5-2B-Base/
  models/Qwen3.5-4B-Base/
  models/Qwen3-4B/
  models/dinov2-base/
  models/dinov2-large/
  models/flux2/FLUX.2-klein-base-4B/
  models/flux2/FLUX.2-dev/ae.safetensors
  models/init/
  data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot/
  data/robotwin2.0/robotwin2.0/
  data/robotwin2.0/dataset_stats.json
  cache/
  runs/
```

The Hugging Face dataset repositories distribute `.tar.gz` archives: extract the four LIBERO archives into `data/libero_mujoco3.3.2/`, and join the RoboTwin parts before extracting (`cat robotwin2.0.tar.gz.part-* | tar -xzf -` inside `data/robotwin2.0/`), retaining `meta/`, `data/` and `videos/`. Check the resulting directory layout against the original backend READMEs. Do not concatenate train and held-out episodes or substitute a different robot dataset. Keep each trained model's own `dataset_stats.json` for evaluation.

## Action initialization

```bash
python tools/init_actions.py wan --dry-run
python tools/init_actions.py wan
python tools/init_actions.py ovis --action-dim 7
python tools/init_actions.py flux --action-dim 14
```

These call the preserved upstream initialization scripts. CPU initialization needs substantial RAM. DINO and native Qwen action experts initialize compatible blocks directly from their own backbone.

## Text and foresight caches

```bash
python tools/jam.py text-cache wan-jam
python tools/jam.py text-cache dino-b-jam
python tools/jam.py text-cache robotwin-flux-jam
python tools/jam.py cache wan-jam
```

Ovis and Qwen3.5 have no separate UMT5/Qwen3 instruction-cache step. Wan target extraction is resumable and follows the same sample indices and target computation as training. To train with the extracted targets:

```bash
python tools/jam.py train wan-jam \
  --override 'model.foresight.target_cache_roots=[/path/to/cache/wan-jam/targets]'
```

Online frozen-anchor training remains available for all families. Ovis's cached timing variant is an explicit **timing emulation**, not a reusable training-target dataset. Never use `IMAGEWAM_FORESIGHT_CACHE_EMULATE=1` to train or report a policy.
