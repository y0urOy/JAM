# Paper-to-code map

Table files reproduce the paper's tables, with its numbers and rounding; recipes give the implementation and settings behind each row.

| Table source / label | Reproduction entry |
|---|---|
| `paper/tables/budget.tex` (tab:budget) | `wan-budget10k-generation`, `wan-budget10k-pretrained-anchor`, `wan-generation`, `wan-pretrained-anchor` |
| `paper/tables/depth.tex` (tab:depth) | `wan-action-only`, `wan-generation`, `wan-depth-14`, `wan-depth-16`, `wan-depth-18`, `wan-depth-20` |
| `paper/tables/factorial.tex` (tab:factorial, tab:imagewamarchmatched) | `wan-factorial-pretrained-pretrained`, `wan-factorial-pretrained-random`, `wan-factorial-random-pretrained`, `wan-factorial-random-random`, `ovis-generation`, `ovis-jam`, `dino-b-jam`, `dino-l-jam`, `qwen-jam` |
| `paper/tables/factorial_full.tex` (tab:factorialfull) | `wan-factorial-pretrained-pretrained`, `wan-factorial-pretrained-random`, `wan-factorial-random-pretrained`, `wan-factorial-random-random` |
| `paper/tables/head.tex` (tab:head) | `wan-action-only`, `wan-generation`, `wan-head-transformer-3072`, `wan-head-transformer-1536`, `wan-head-transformer-768`, `wan-head-mlp-3072`, `wan-head-mlp-1536` |
| `paper/tables/head_dims.tex` (tab:headdims) | `wan-jam`, `ovis-jam`, `qwen-jam`, `dino-b-jam`, `dino-l-jam`, `robotwin-flux-jam` |
| `paper/tables/lambda_weight.tex` (tab:lambdaweight) | `wan-action-only`, `wan-generation`, `wan-weight-0.4`, `wan-weight-0.5`, `wan-weight-0.6`, `wan-weight-0.7` |
| `paper/tables/libero.tex` (tab:libero) | `wan-generation`, `wan-jam`, `ovis-generation`, `ovis-jam`, `qwen-jam` |
| `paper/tables/main_liberoplus.tex` (tab:mainplus) | `ovis-generation`, `wan-generation`, `wan-jam`, `ovis-jam`, `qwen-jam` |
| `paper/tables/main_pair_avg.tex` (tab:mainrobotwin, tab:mainlibero) | `ovis-generation`, `wan-generation`, `wan-jam`, `ovis-jam`, `qwen-jam`, `robotwin-wan-jam`, `robotwin-ovis-jam`, `robotwin-flux-jam`, `robotwin-qwen-jam` |
| `paper/tables/multiseed.tex` (tab:multiseed) | `wan-jam`, `ovis-jam` (three runs each) |
| `paper/tables/objectives.tex` (tab:objectives) | `wan-generation`, `wan-action-only`, `wan-pretrained-anchor`, `wan-random-anchor`, `wan-jam`, `wan-readout-0`, `wan-readout-012`, `ovis-generation`, `ovis-action-only`, `ovis-jam`, `ovis-readout-0`, `ovis-readout-01`, `ovis-random-anchor`, `ovis-vae-anchor` |
| `paper/tables/paired_statistics.tex` (tab:pairedstats) | `wan-generation`, `wan-jam`, `ovis-generation`, `ovis-jam` |
| `paper/tables/qwen_action_only.tex` (tab:qwenactiononly) | `qwen-jam`, `qwen-action-only`, `dino-l-jam`, `dino-l-action-only` |
| `paper/tables/robotwin_main.tex` (tab:robotwinmain) | `robotwin-wan-jam`, `robotwin-ovis-jam`, `robotwin-flux-jam`, `robotwin-qwen-jam` |
| `paper/tables/shared_weights.tex` (tab:sharedweights) | `qwen-jam`, `qwen-shared-2b`, `qwen-shared-4b` |
| `paper/tables/target_interfaces.tex` (tab:targetinterfaces) | `wan-pretrained-anchor`, `wan-jam`, `ovis-jam`, `qwen-jam`, `dino-b-jam`, `dino-l-jam`, `robotwin-flux-jam` |
| `paper/tables/teacher_full.tex` (tab:teacherfull) | `wan-action-only`, `wan-generation`, `wan-pretrained-anchor`, `wan-ema-anchor` |
| `paper/tables/tab_costbreakdown.tex` (tab:costbreakdown) | Analysis/reference: see docs/ANALYSES.md and exact table caption |

Figure 1/2 layout and editor generators: `paper/figure_source/`. Install `python-pptx`, `cairosvg`, `cairocffi` plus system Cairo/fonts, then use `build_restored_figure_editors.py --output-dir /path/to/output --figure1-relayout` or `--figure2-relayout`. Export through PowerPoint/LibreOffice; font/rendering differences are platform dependent.

Paired rollout/attention/readout figure definitions: `paper/figure_definitions/`. Actual extraction and rendering tools are under `backends/multimodal/scripts/{probe,attn_vis}` and the Wan RoboTwin attention adapter. External models/data and probe readout weights are separate assets.

The 48 recipes reproduce the recorded task configurations and paper settings. They are checked as configurations; they were not retrained for this release.
