# Validation

## Checked

- **CPU tests.** `pytest tests` runs in GitHub Actions on every push. It checks that all 48 recipes compose without
  missing fields and match their recorded configuration fingerprints; that the launcher keeps a recipe's global batch
  when the GPU layout changes and rejects evaluation on a mismatched robot; that the fixed 1,506-task LIBERO-Plus list
  is complete and unique; that result summaries reject missing or duplicate episodes; that paired statistics reject
  mismatched episodes; and that legacy run, checkpoint and import names still resolve.
- **Small-model checks on CPU** (`backends/wan/scripts/{smoke_test_foresight,test_foresight_cache,test_qwen_foresight_offsets}.py`,
  `backends/multimodal/scripts/test_*.py`): foresight heads and targets, cached versus online targets, and the native
  Qwen3.5 and DINOv2 action experts, each in its own environment profile. They use tiny models, not the trained
  checkpoints.
- **Checkpoints.** The three LIBERO checkpoints on Hugging Face load on CPU, and their sizes and SHA256 match
  `reproduction/libero_checkpoints.json`.
- **Files.** `tools/verify_manifest.py` checks the size and SHA256 of every file listed in `FILE_MANIFEST.json` (also in
  CI); `SHA256SUMS` lists the same hashes for `sha256sum -c`. `website/src/build.py --check` checks the project page
  against its sources.

## Not checked

- Training, GPU inference and benchmark rollouts from this tree. The checks above do not reproduce success rates.
- A fresh CUDA installation and simulator rendering on a new machine.
- Paper results whose weights or raw results are not released; see `reproduction/artifact_status.json`.
