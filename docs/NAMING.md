# JAM implementation names

The paper names the policies **JAM (Wan)**, **JAM (Ovis-U1)**, **JAM (Qwen3.5-2B)**, **JAM (DINOv2-B/L)** and **JAM (FLUX.2)**. JAM retains the visual backbone and action expert at deployment; future features supervise training without image or video generation.

| Component | Canonical location / name |
|---|---|
| Wan backend | `backends/wan`, Python package `jam_wan` |
| Ovis-U1, Qwen3.5, DINOv2, FLUX.2 | `backends/multimodal`, Python package `jam_multimodal` |
| Wan policy | `jam_wan.models.wan22.jam_wan.JAMWan` |
| Ovis-U1 / DINOv2 / FLUX.2 policy | `jam_multimodal.models.backbones.jam.JAMPolicy`; DINOv2 replaces the visual stream and action expert through `jam_multimodal.models.backbones.external_ssl` |
| Qwen3.5 policy | `jam_multimodal.models.backbones.jam_qwen35.JAMQwen35` |
| Model factories | `create_jam_wan`, `create_jam_ovis_u1` (Ovis-U1 and DINOv2), `create_jam_qwen35`, `create_jam_flux2_klein` |
| Dataset processors | `JAMProcessor` in each backend |
| RoboTwin adapters | `experiments/robotwin/jam_policy` in each backend |
| Train / evaluate | `python tools/jam.py train-eval RECIPE` |

The ten recipe IDs (`wan-jam`, `ovis-jam`, `qwen-jam`, `dino-b-jam`, `dino-l-jam`, `flux-jam-libero` and the four `robotwin-*-jam`) are stable. All ten run in the one environment that `tools/bootstrap.py` installs.

## Existing runs and checkpoints

- The migration changes namespaces, file paths and display names. Network dimensions, parameter/state-dict keys, losses, schedules and the ten named recipes' scientific settings are preserved. `config_fingerprints.json` records each recipe's resolved settings, excluding output/logging names.
- Historical `fastwam` / `imagewam` imports, model classes and resolved Hydra targets remain available through compatibility aliases for the released modules. Old CLI backend names and environment-variable prefixes are accepted; explicit `JAM_*` settings take precedence.
- `eval --run-dir` permits known namespace-only differences in `run.json` but still rejects scientific recipe drift and changed configuration/statistics files. Original records are never rewritten.
- Direct Hydra task/model file names have changed; [the migration map](../reproduction/naming.json) lists the replacements of the files that this release contains. Use stable recipe IDs for new runs. For an archived run with custom paths, use its pinned source or apply that explicit map.
- Saved checkpoint format tags and predictor discriminator strings retain their historical spelling. They identify binary layouts; renaming them would break loading. The three LIBERO weight files, statistics and recorded results are unchanged. Model cards pin the reproduction-source revision that loads each checkpoint.

FastWAM and ImageWAM remain the names of the upstream projects. Their citations, licenses and original artifact IDs are preserved.
