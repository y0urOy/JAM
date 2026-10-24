# Probing, attention and compute

Run each analysis in its matching backbone environment. The tools below read frozen run directories containing `config.yaml`, normalization statistics and the selected weights.

## Future-frame probing

```bash
cd backends/multimodal
python scripts/probe/future_probe.py --run /path/to/run --family yak \
  --out /path/to/probe --cache-dir /path/to/scratch
python scripts/probe/readout_diff.py --run /path/to/run --family yak \
  --probe-out /path/to/probe --out /path/to/readout --cache-dir /path/to/scratch
python scripts/probe/assemble_probe.py --runs native=/path/to/native jam=/path/to/jam --out-dir /path/to/panels
```

The probe code keeps training and held-out episodes disjoint. The two-layer per-token MLP probes reconstruct current and 16-step future frames; they do not enter the policy. Report aggregate PSNR separately from selected qualitative examples. Probe readout weights are not released (see `reproduction/artifact_status.json`); the paper's PSNR values are reported results.

The historical selected JAM `[0]` signed-change panel applies a 0.5 scale to positive red readout values. The video renderer exposes this as `--attenuate off0:0.5`. Use that option only to reproduce the historical rendering; use identical scales across models for a symmetric new visualization. This does not change raw probe predictions or PSNR.

## Action-to-visual attention

```bash
cd backends/multimodal
python scripts/attn_vis/make_manifest_many.py --help
python scripts/attn_vis/attn_vis.py --run /path/to/run --family yak \
  --manifest /path/to/manifest.json --out /path/to/attention --steps 10 --seed 42
python scripts/attn_vis/group_shares.py /path/to/shares.json /path/to/captures:yak,dino,dinol,qwen
python scripts/attn_vis/make_figure.py --help
```

Families are `yak`, `qwen`, `dino`; the same source also retains the older VideoMAE analysis. Qwen's softmax capture covers its six full-attention action blocks, not its recurrent GatedDeltaNet layers. The paper averages heads, 16 action queries, ten denoising steps and captured layers. Normalize per view, subtract each policy's 100-reference-frame mean, retain positive excess, divide by the 99th percentile, clip and apply power 0.6. Difference maps use the transformed maps and suppress magnitudes below 0.25. These are descriptive attention visualizations.

RoboTwin attention adapters are `backends/multimodal/scripts/attn_vis/attn_vis_robotwin.py` and `backends/wan/scripts/attn_vis/attn_vis_robotwin_wan.py`. Original method-figure layout/generation code is in `paper/figure_source/`.

## Training cost

```bash
python tools/jam.py profile wan-jam --dry-run
python tools/profile_ovis.py online --dry-run
python tools/profile_ovis.py cached-emulated --dry-run
python paper/figure_source/scripts/imagewam_cost_accounting.py
```

The Wan script uses actual model forward/backward passes and a pure CUDA-activity profiler anchor. Ovis uses the preserved `IMAGEWAM_PROFILE_GPU_STEPS` instrumentation. The cached Ovis variant reuses first-batch target features solely to emulate timing; it is not valid model training. Record GPU model, batch, activation checkpointing and the operations covered. Cached FLOPs/timings exclude offline target extraction and are not an end-to-end wall-clock speedup.

The paper's cost table is `paper/tables/tab_costbreakdown.tex`.
