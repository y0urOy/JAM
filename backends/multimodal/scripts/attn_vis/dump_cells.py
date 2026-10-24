"""Export per-cell PNGs (raw view + one overlay per family) so the paper can lay out the
attention grid in TikZ with document fonts.  Mirrors make_figure.py (layer mean, alpha 0.55, 99th-pct clip).

Usage: python dump_cells.py --root OUTROOT --order yak,dino,dinol,videomae,qwen --out CELLDIR
Files: {A|B}_{agent|wrist}_{image|<family>}.png
"""
import argparse, glob, os
import numpy as np
from PIL import Image
from make_figure import overlay, pick

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen")
ap.add_argument("--out", required=True); ap.add_argument("--layer", default="mean"); ap.add_argument("--alpha", type=float, default=0.55)
args = ap.parse_args()
fams = args.order.split(",")
tags = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(args.root, fams[0], "*.npz")))]
tags = sorted(tags, key=lambda t: 0 if "libero_goal" in t else 1)  # A = bowl on stove (goal), B = mug on plate (long)
os.makedirs(args.out, exist_ok=True)
for j, tag in enumerate(tags):
    task = chr(65 + j)
    base = np.load(os.path.join(args.root, fams[0], tag + ".npz"), allow_pickle=True)
    views = {"agent": base["agent"], "wrist": base["wrist"]}
    for v, img in views.items():
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(os.path.join(args.out, f"{task}_{v}_image.png"))
    for fam in fams:
        d = np.load(os.path.join(args.root, fam, tag + ".npz"), allow_pickle=True)
        m = pick(d["maps"], d["layers"], args.layer)
        for k, v in enumerate(["agent", "wrist"]):
            out = overlay(np.asarray(views[v]), m[k], args.alpha)
            Image.fromarray((out * 255).round().astype(np.uint8)).save(os.path.join(args.out, f"{task}_{v}_{fam}.png"))
    print(task, tag, "views", {v: np.asarray(i).shape for v, i in views.items()})
print("wrote", len(os.listdir(args.out)), "cells to", args.out)
