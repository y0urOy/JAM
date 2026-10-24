#!/usr/bin/env python3
"""Assemble future-probe outputs of several policies into one summary table and figure panel."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=probe_out_dir ... (first one supplies copy-current)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preview-dir", default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    return ap.parse_args()


def mid_layer(summary):
    return [l for l in summary["layers"] if l != "final"][0]


def fmt(a):
    return (f"{a['psnr']:.2f} | {a['ssim']:.3f} | {a['changed_l1_vs_future']:.3f} | {a['changed_l1_vs_current']:.3f} | "
            f"{100 * a['changed_frac_closer_future']:.1f}% | {a['static_l1_vs_future']:.3f} | {a['n']}")


def load_png(p):
    return np.asarray(Image.open(p).convert("RGB"))


def redblue(pred, cur, fut, thr=0.10, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    p, c, f = (x.astype(np.float32) / 255.0 for x in (pred, cur, fut))
    diff = np.abs(c - f).mean(-1)
    mask = gaussian_filter(diff, sigma) > thr
    closer = np.abs(p - f).mean(-1) < np.abs(p - c).mean(-1)
    vis = f * 0.55 + 0.45 * 0.5
    vis[mask & closer] = [0.85, 0.15, 0.15]
    vis[mask & ~closer] = [0.15, 0.3, 0.9]
    return (vis * 255).astype(np.uint8)


def main():
    args = parse()
    runs = []
    for item in args.runs:
        name, d = item.split("=", 1)
        d = Path(d)
        runs.append((name, d, json.loads((d / "summary.json").read_text())))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["| model | predictor | PSNR | SSIM | changed L1 vs future | changed L1 vs current | closer to future | static L1 | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    first = runs[0][2]
    lines.append(f"| (all) | copy current frame | {fmt(first['predictors']['copy_current']['all'])} |")
    per_suite = {}
    for name, d, s in runs:
        ml = mid_layer(s)
        for key, label in ((f"probe_future_L{ml}", f"probe, layer {ml}, future"), ("probe_future_Lfinal", "probe, final layer, future"),
                           (f"probe_current_L{ml}", f"probe, layer {ml}, current (reference)"), ("generated", "own generation")):
            if key in s["predictors"]:
                lines.append(f"| {name} | {label} | {fmt(s['predictors'][key]['all'])} |")
        per_suite[name] = {g: v for g, v in s["predictors"][f"probe_future_L{ml}"].items()}
    suites = [g for g in first["predictors"]["copy_current"] if g != "all"]
    lines += ["", "Per-suite PSNR of the mid-layer future probe (copy-current in parentheses):", "",
              "| model | " + " | ".join(suites) + " |", "|---|" + "---|" * len(suites)]
    for name, d, s in runs:
        lines.append(f"| {name} | " + " | ".join(f"{per_suite[name][g]['psnr']:.2f} ({first['predictors']['copy_current'][g]['psnr']:.2f})" for g in suites) + " |")
    if "generated" in first["predictors"]:
        g0 = first["predictors"]["generated"]
        lines.append(f"| {runs[0][0]} generated | " + " | ".join(f"{g0[g]['psnr']:.2f}" for g in suites) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({name: s for name, _, s in runs}, open(out / "summary_all.json", "w"), indent=1)

    # --- consistency of the held-out visual subset across runs
    idx = {name: json.loads((d / "vis" / "index.json").read_text()) for name, d, _ in runs}
    tags = [e["tag"] for e in idx[runs[0][0]]]
    for name, entries in idx.items():
        keys_a = [(e["index"], e.get("episode_index"), e.get("frame_index")) for e in idx[runs[0][0]]]
        keys_b = [(e["index"], e.get("episode_index"), e.get("frame_index")) for e in entries]
        if keys_a != keys_b:
            print(f"WARNING held-out vis subset differs for {name}: {keys_b[:3]} vs {keys_a[:3]}")
    for tag in tags:
        ref = load_png(runs[0][1] / "vis" / f"{tag}_future.png").astype(np.int16)
        for name, d, _ in runs[1:]:
            p = d / "vis" / f"{tag}_future.png"
            if p.is_file():
                diff = np.abs(load_png(p).astype(np.int16) - ref).max()
                if diff > 8:
                    print(f"WARNING future frame mismatch {name} {tag}: max abs diff {diff}")

    # --- panels
    cols = [("current", runs[0][1], "current"), ("future", runs[0][1], "future")]
    if (runs[0][1] / "vis" / f"{tags[0]}_generated.png").is_file():
        cols.append((f"{runs[0][0]} generation", runs[0][1], "generated"))
    for name, d, s in runs:
        cols.append((f"{name} probe L{mid_layer(s)}", d, f"probe_future_L{mid_layer(s)}"))
    sc = args.scale
    W, H = int(448 * sc), int(224 * sc)
    pad, header = 4, 18
    font = ImageFont.load_default()
    for panel_name, use_rb in (("panel", False), ("panel_redblue", True)):
        canvas = Image.new("RGB", (len(cols) * (W + pad), header + len(tags) * (H + pad)), "white")
        dr = ImageDraw.Draw(canvas)
        for ci, (label, _, _) in enumerate(cols):
            dr.text((ci * (W + pad) + 2, 2), label, fill="black", font=font)
        for ri, tag in enumerate(tags):
            cur = load_png(runs[0][1] / "vis" / f"{tag}_current.png")
            fut = load_png(runs[0][1] / "vis" / f"{tag}_future.png")
            for ci, (label, d, key) in enumerate(cols):
                p = d / "vis" / f"{tag}_{key}.png"
                if not p.is_file():
                    continue
                img = load_png(p)
                if use_rb and key not in ("current", "future"):
                    img = redblue(img, cur, fut)
                im = Image.fromarray(img).resize((W, H), Image.BILINEAR)
                canvas.paste(im, (ci * (W + pad), header + ri * (H + pad)))
        canvas.save(out / f"{panel_name}.png")
        if args.preview_dir:
            Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(out / f"{panel_name}.png", Path(args.preview_dir) / f"future_probe_{panel_name}.png")
    print("ASSEMBLE_DONE", out)


if __name__ == "__main__":
    main()
