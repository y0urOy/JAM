# Validation

## Checked

- **CPU tests.** `pytest tests` runs in GitHub Actions on every push. It checks that the ten JAM recipes compose without
  missing fields and match their recorded configuration fingerprints; that every training, cache, initialization and
  evaluation entry point they use exists, and that their LIBERO, LIBERO-Plus and RoboTwin evaluation configurations
  compose; that the launcher keeps a recipe's global batch when the GPU layout changes and rejects evaluation on a
  mismatched robot; that `tools/bootstrap.py` installs every requirement against the pinned constraints; that the
  downloadable assets are exactly the ones the recipes use; that the fixed 1,506-task LIBERO-Plus list is complete and
  unique; that result summaries reject missing or duplicate episodes; and that backend imports and legacy run,
  checkpoint and import names still resolve.
- **Small-model checks on CPU** (`backends/wan/scripts/{smoke_test_foresight,test_foresight_cache}.py`,
  `backends/multimodal/scripts/test_*.py`): foresight heads and targets, cached versus online targets, the Ovis-U1 and
  FLUX.2 foresight paths, and the native Qwen3.5 and DINOv2 action experts, in the shared environment. They
  use tiny models; the two Qwen3.5 checks also read the Qwen3.5-2B-Base configuration and processor from
  `artifacts/models/Qwen3.5-2B-Base` (or `$JAM_MODEL_ROOT/Qwen3.5-2B-Base`).
- **Checkpoints.** The three LIBERO checkpoints on Hugging Face load on CPU, and their sizes and SHA256 match
  `reproduction/libero_checkpoints.json`.
- **Shared runtime.** A fresh installation was checked on a GH200 against the original backbone environments with
  real models, fixed inputs and saved initial weights. Deterministic policy outputs, features and losses match;
  gradient comparisons account for CUDA reduction and kernel-selection variability. Qwen comparisons replay the
  same FLA autotune choices. Transformers 5 changes DINO's CPU RNG consumption while loading pretrained weights,
  so a fresh DINO run with the same seed initializes 28 new tensors differently from Transformers 4; loading the
  same saved initialization removes that difference.
- **Optimizer updates.** Real Wan, FLUX.2 and Qwen3.5 policies completed two ZeRO-1 AdamW updates in both the
  original and shared environments, with two independent runs per environment. Wan and FLUX.2 used eight GPUs;
  Qwen3.5 used one. Parameters, gradients and FP32 optimizer states remained finite, and second moments were
  nonnegative. These short checks used a fixed base learning rate and one sample per rank; independent BF16
  updates vary, so they do not establish bitwise training trajectories or full-run benchmark equivalence.
- **Files.** `tools/verify_manifest.py` checks the size and SHA256 of every file listed in `FILE_MANIFEST.json` (also in
  CI); `SHA256SUMS` lists the same hashes for `sha256sum -c`. `website/src/build.py --check` checks the project page
  against its sources.
