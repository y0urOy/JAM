# Data and models

Download references and revisions are in [assets.json](../reproduction/assets.json). The trained LIBERO policies are on [Hugging Face](https://huggingface.co/y0urOy/JAM); see [libero_checkpoints.json](../reproduction/libero_checkpoints.json).

| Recipes | Assets |
|---|---|
| `libero-wan-jam`, `robotwin-wan-jam` | `wan`, `wan-tokenizer` |
| `libero-ovis-jam`, `robotwin-ovis-jam` | `ovis` |
| `libero-flux-jam`, `robotwin-flux-jam` | `flux`, `flux-ae`, `qwen3` |
| `libero-qwen-jam`, `robotwin-qwen-jam` | `qwen-2b` |
| `libero-dino-b-jam` / `libero-dino-l-jam` | `dino-b` / `dino-l`, `qwen3`, `ovis` |
| LIBERO / RoboTwin 2.0 data | `libero-data` / `robotwin-data` |
| LIBERO-Plus / RoboTwin 2.0 simulator assets | `libero-plus-assets` / `robotwin-assets` (see [INSTALL.md](INSTALL.md)) |

`flux-ae` is the FLUX.2-dev autoencoder, a gated Hugging Face repository under the FLUX non-commercial license: accept
its terms on Hugging Face and log in (`HF_TOKEN` or the Hugging Face CLI) before downloading it.

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
  models/Qwen3-4B/
  models/dinov2-base/
  models/dinov2-large/
  models/flux2/FLUX.2-klein-base-4B/
  models/flux2/FLUX.2-dev/ae.safetensors
  models/init/
  data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot/
  data/robotwin2.0/robotwin2.0/
  data/robotwin2.0/robotwin2.0/nonidle_ranges.json
  data/robotwin2.0/dataset_stats.json
  cache/
  runs/
```

The Hugging Face dataset repositories distribute `.tar.gz` archives: extract the four LIBERO archives into `data/libero_mujoco3.3.2/`, and join the RoboTwin parts before extracting (`cat robotwin2.0.tar.gz.part-* | tar -xzf -` inside `data/robotwin2.0/`), retaining `meta/`, `data/` and `videos/`. Do not concatenate train and held-out episodes or substitute a different robot dataset. Keep each trained model's own `dataset_stats.json` for evaluation.

The Ovis-U1, Qwen3.5 and FLUX.2 RoboTwin recipes train on the non-idle frames of each episode, as ImageWAM does: idle
pauses of five or more steps are removed from the sampled windows and from the timeline of each 16-step action chunk
and its future frame. Build the frame ranges once after extracting the data:

```bash
python backends/multimodal/scripts/data/compute_robotwin_nonidle_ranges.py artifacts/data/robotwin2.0/robotwin2.0 \
  --output artifacts/data/robotwin2.0/robotwin2.0/nonidle_ranges.json
```

Its summary should report 27,500 episodes, 6,075,103 steps and 5,424,603 kept steps, and training then logs
`kept 5368277/6011575 frames`. The file records the absolute dataset path, so its checksum depends on where the data
lives. The RoboTwin Wan recipe trains on all frames.

## Action initialization

Run these commands in the environment that `tools/bootstrap.py` installs:

```bash
python tools/init_actions.py wan --dry-run
python tools/init_actions.py wan                    # libero-wan-jam, robotwin-wan-jam
python tools/init_actions.py ovis --action-dim 7    # libero-ovis-jam
python tools/init_actions.py ovis --action-dim 14   # robotwin-ovis-jam
python tools/init_actions.py flux --action-dim 7    # libero-flux-jam
python tools/init_actions.py flux --action-dim 14   # robotwin-flux-jam
```

These call the preserved upstream initialization scripts. Ovis-U1 and FLUX.2 run on the CPU and need substantial RAM.
The Wan initialization runs in BF16 on a CUDA GPU, as the original was made, and needs about 26 GB of GPU memory and
40 GB of host RAM. It reproduces the original bit for bit; its log reports `copied=300, interpolated=520, skipped=4`, and
the SHA256 over its archive records (the file checksum also depends on the file name) is
`949c93e10154977b0f9906b37871049e07e0715506368d57203eaa014bf7ecc7`:

```bash
python -c "import hashlib,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); h=hashlib.sha256(); [h.update(z.read(n)) for n in z.namelist()]; print(h.hexdigest())" artifacts/models/init/wan_action.pt
```
DINOv2 and native Qwen3.5 action experts initialize compatible blocks directly from their own backbone.

## Text and foresight caches

Wan uses a UMT5 instruction cache; FLUX.2 and DINOv2 use a Qwen3-4B instruction cache:

```bash
python tools/jam.py text-cache libero-wan-jam
python tools/jam.py text-cache robotwin-wan-jam
python tools/jam.py text-cache libero-flux-jam
python tools/jam.py text-cache robotwin-flux-jam
python tools/jam.py text-cache libero-dino-b-jam
python tools/jam.py text-cache libero-dino-l-jam
python tools/jam.py cache libero-wan-jam
```

The Qwen3-4B caches hold 512 tokens (`qwen3_4b_len512`) for the LIBERO FLUX.2 and DINOv2 recipes and 128 tokens (`robotwin/qwen3_4b_len128`) for `robotwin-flux-jam`, as in the `config.yaml` shipped with the released ImageWAM FLUX.2 RoboTwin checkpoint (the upstream repository task config uses 512); FLUX.2 evaluation encodes instructions online at 512 tokens. Ovis-U1 and Qwen3.5 have no separate UMT5/Qwen3 instruction-cache step. The LIBERO-Plus evaluation of the DINOv2 recipes needs extra cache entries; see [EVALUATION.md](EVALUATION.md). Wan target extraction (`cache`) is optional and resumable, and follows the same sample indices and target computation as training. To train with the extracted targets:

```bash
python tools/jam.py train libero-wan-jam \
  --override 'model.foresight.target_cache_roots=[/path/to/cache/libero-wan-jam/targets]'
```

Without a target cache, every family computes its frozen foresight targets online during training.
