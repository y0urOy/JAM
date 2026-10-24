"""Per-layer strips: for each family and task, agent+wrist overlays at every captured layer.

Usage: python layer_strip.py --root OUTROOT --order yak,dino,dinol,videomae,qwen --out DIR
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_figure import overlay, LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for fam in a.order.split(","):
        for f in sorted(glob.glob(os.path.join(a.root, fam, "*.npz"))):
            d = np.load(f, allow_pickle=True); maps = d["maps"].mean(axis=1)  # [L, V, gh, gw]
            L = maps.shape[0]
            fig, axes = plt.subplots(2, L, figsize=(1.6 * L, 3.4))
            if L == 1: axes = axes.reshape(2, 1)
            for l in range(L):
                axes[0, l].imshow(overlay(d["agent"], maps[l, 0])); axes[1, l].imshow(overlay(d["wrist"], maps[l, 1]))
                axes[0, l].set_title(f"L{int(d['layers'][l])}", fontsize=7)
                for r in (0, 1): axes[r, l].set_xticks([]); axes[r, l].set_yticks([])
            fig.suptitle(f"{LABELS.get(fam, fam)}: {d['instruction']}", fontsize=8)
            plt.tight_layout(); out = os.path.join(a.out, f"{fam}_{os.path.basename(f)[:-4]}_layers.png")
            plt.savefig(out, dpi=150); plt.close(fig); print("saved", out)


if __name__ == "__main__":
    main()
