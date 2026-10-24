"""Assemble attention overlays into a grid: rows = policies, columns = task x view.

Usage: python make_figure.py --root OUTROOT --order yak,dino,dinol,videomae,qwen --out fig.png [--layer mean|last|N]
Each family directory under OUTROOT holds one npz per manifest record produced by attn_vis.py.
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

LABELS = {"yak": "Yak18 JAM", "dino": "DINOv2-B JAM", "dinol": "DINOv2-L JAM", "videomae": "VideoMAE-B JAM", "qwen": "Qwen3.5-2B JAM"}


def overlay(img, amap, alpha=0.55, size=None, clip_pct=99.0):
    h, w = img.shape[:2]
    hi = np.percentile(amap, clip_pct)
    m = np.clip(amap - amap.min(), 0, None); m = np.clip(m / (hi - amap.min() + 1e-8), 0, 1)
    heat = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)).astype(np.float32) / 255.0
    color = cm.get_cmap("jet")(heat)[..., :3]
    out = (1 - alpha) * img.astype(np.float32) / 255.0 + alpha * color
    return np.clip(out, 0, 1)


INCLUDE_AUX = False  # layer ids >= 100 are Yak's image-self attention (action -> image only)


def pick(maps, layers, how):
    # maps: [L, S, V, gh, gw]; average over denoising steps first.
    keep = np.asarray(layers) < 100 if not INCLUDE_AUX else np.ones(len(layers), dtype=bool)
    maps = maps[keep]; layers = np.asarray(layers)[keep]
    m = maps.mean(axis=1)
    if how == "mean": return m.mean(axis=0)
    if how == "last": return m[-1]
    return m[layers.tolist().index(int(how))]


def main_transposed(args, fams, tags):
    """Rows = task/view, columns = Image + policies (wide layout for a text-width figure)."""
    ncol = 1 + len(fams); nrow = 2 * len(tags)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.75 * ncol, 1.75 * nrow), gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for j, tag in enumerate(tags):
        base = np.load(os.path.join(args.root, fams[0], tag + ".npz"), allow_pickle=True)
        agent, wrist = base["agent"], base["wrist"]
        axes[2 * j, 0].imshow(agent); axes[2 * j + 1, 0].imshow(wrist)
        for i, fam in enumerate(fams):
            d = np.load(os.path.join(args.root, fam, tag + ".npz"), allow_pickle=True)
            m = pick(d["maps"], d["layers"], args.layer)
            axes[2 * j, i + 1].imshow(overlay(agent, m[0], args.alpha)); axes[2 * j + 1, i + 1].imshow(overlay(wrist, m[1], args.alpha))
    for i, name in enumerate(["Image"] + [LABELS.get(f, f) for f in fams]):
        axes[0, i].set_title(name, fontsize=9)
    row_labels = []
    for j in range(len(tags)):
        row_labels += [f"Task {chr(65 + j)}, agent view", f"Task {chr(65 + j)}, wrist view"]
    for r, lab in enumerate(row_labels):
        axes[r, 0].set_ylabel(lab, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(args.out, dpi=200, bbox_inches="tight"); print("saved", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen")
    ap.add_argument("--out", required=True); ap.add_argument("--layer", default="mean"); ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--include-aux", action="store_true"); ap.add_argument("--transpose", action="store_true")
    args = ap.parse_args()
    global INCLUDE_AUX; INCLUDE_AUX = bool(args.include_aux)
    fams = args.order.split(",")
    files = sorted(glob.glob(os.path.join(args.root, fams[0], "*.npz")))
    tags = [os.path.basename(f)[:-4] for f in files]
    tags = sorted(tags, key=lambda t: 0 if "libero_goal" in t else 1)  # Task A = bowl on stove, Task B = mug on plate
    if args.transpose:
        return main_transposed(args, fams, tags)
    ncol = 2 * len(tags); nrow = 1 + len(fams)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.9 * nrow + 0.4), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    stats = {}
    for j, tag in enumerate(tags):
        base = np.load(os.path.join(args.root, fams[0], tag + ".npz"), allow_pickle=True)
        agent, wrist = base["agent"], base["wrist"]
        axes[0, 2 * j].imshow(agent); axes[0, 2 * j + 1].imshow(wrist)
        import textwrap
        axes[0, 2 * j].set_title("\n".join(textwrap.wrap('"' + str(base["instruction"]) + '"', 46)), fontsize=8, loc="left", style="italic")
        for i, fam in enumerate(fams):
            d = np.load(os.path.join(args.root, fam, tag + ".npz"), allow_pickle=True)
            m = pick(d["maps"], d["layers"], args.layer)  # [V, gh, gw]
            axes[i + 1, 2 * j].imshow(overlay(agent, m[0], args.alpha))
            axes[i + 1, 2 * j + 1].imshow(overlay(wrist, m[1], args.alpha))
            stats.setdefault(fam, {})[tag] = {"agent_mass": float(m[0].sum() / m.sum()), "wrist_mass": float(m[1].sum() / m.sum()),
                                              "agent_peak_rc": [int(x) for x in np.unravel_index(m[0].argmax(), m[0].shape)],
                                              "wrist_peak_rc": [int(x) for x in np.unravel_index(m[1].argmax(), m[1].shape)],
                                              "agent_entropy": float(-(m[0] / m[0].sum() * np.log(m[0] / m[0].sum() + 1e-12)).sum()),
                                              "wrist_entropy": float(-(m[1] / m[1].sum() * np.log(m[1] / m[1].sum() + 1e-12)).sum()),
                                              "grid": list(m.shape[1:])}
    for i, fam in enumerate(["Image"] + fams):
        axes[i, 0].set_ylabel(LABELS.get(fam, fam), fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(args.out, dpi=200, bbox_inches="tight"); print("saved", args.out)
    json.dump(stats, open(os.path.splitext(args.out)[0] + "_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
