# JAM implementation names

The paper names the policies **JAM (Wan)**, **JAM (Ovis-U1)**, **JAM (Qwen3.5-2B)**, **JAM (DINOv2-B/L)** and **JAM (FLUX.2)**. JAM retains the visual backbone and action expert at deployment; future features supervise training without image or video generation.

| Component | Canonical location / name |
|---|---|
| Wan backend | `backends/wan`, Python package `jam_wan` |
| Ovis-U1, Qwen3.5, DINOv2, FLUX.2 | `backends/multimodal`, Python package `jam_multimodal` |
| Wan policy | `jam_wan.models.wan22.jam_wan.JAMWan` |
| Ovis-U1 / FLUX.2 policy | `jam_multimodal.models.backbones.jam.JAMPolicy` |
| Qwen3.5 policy | `jam_multimodal.models.backbones.jam_qwen35.JAMQwen35` |
| Model factories | `create_jam_wan`, `create_jam_ovis_u1`, `create_jam_qwen35`, `create_jam_flux2_klein` |
| Dataset processors | `JAMProcessor` in each backend |
| RoboTwin adapters | `experiments/robotwin/jam_policy` in each backend |
| Train / evaluate | `python tools/jam.py train-eval RECIPE` |

Recipe IDs such as `wan-jam`, `ovis-jam` and `qwen-jam` are stable. Use separate environment profiles because the backbones require different Transformers versions.

## Existing runs and checkpoints

- The migration changes namespaces, file paths and display names. Network dimensions, parameter/state-dict keys, losses, schedules and the 48 named recipes' scientific settings are preserved. `config_fingerprints.json` records each recipe's resolved settings, excluding output/logging names.
- Historical `fastwam` / `imagewam` imports, model classes and resolved Hydra targets remain available through compatibility aliases. Old CLI backend names and environment-variable prefixes are accepted; explicit `JAM_*` settings take precedence.
- `eval --run-dir` permits known namespace-only differences in `run.json` but still rejects scientific recipe drift and changed configuration/statistics files. Original records are never rewritten.
- Direct Hydra task/model file names have changed; [the migration map](../reproduction/naming.json) lists their replacements. Use stable recipe IDs for new runs. For an archived run with custom paths, use its pinned source or apply that explicit map.
- Saved checkpoint format tags and predictor discriminator strings retain their historical spelling. They identify binary layouts; renaming them would break loading. The three LIBERO weight files, statistics and recorded results are unchanged. Model cards pin a matching reproduction-source revision, not a claimed original training revision.

FastWAM and ImageWAM remain the names of the **comparison baselines** and upstream projects. Their citations, licenses, historical environment records and original artifact IDs are preserved. `wan-generation` and `ovis-generation` select the baseline objectives in the shared implementations.

Historical multimodal joint/IDM prototypes whose base implementations were absent are excluded from the runnable configuration tree. They are not any of the 48 named paper recipes; old factory imports now report an explicit unsupported-prototype error. The removed file list is in `naming.json`.
