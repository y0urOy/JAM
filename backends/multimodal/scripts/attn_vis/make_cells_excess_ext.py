"""Figure-5-style cells (surplus over the policy's mean map) for any captured LIBERO/LIBERO-Plus frame.

Same display as make_fig4_cells_excess.py: each frame's layer-mean map is normalized within the view, the
policy's mean map over the reference frames (--mean-root, the 100-frame LIBERO capture) is subtracted, the
result is clipped at zero, scaled to its 99th percentile and raised to GAMMA, then jet-overlaid (alpha 0.55).
The difference cell is the signed surplus of the JAM policy (yak) minus the native policy on a fixed
+-VMAX scale, grey below half a uniform density.  Families missing from --frame-root are skipped.

usage: make_cells_excess_ext.py --mean-root DIR --frame-root DIR --tag TAG --prefix P --out DIR
       [--families native,yak,dino,dinol,qwen] [--views agent,wrist] [--vmax 4] [--gamma 0.6]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from matplotlib import cm
from make_figure import overlay

SZ = 448
T0 = 0.5


def layer_mean(path):
    d = np.load(path, allow_pickle=True)
    keep = np.asarray(d["layers"]) < 100
    m = d["maps"][keep].mean(axis=(0, 1))                       # [V, gh, gw]
    return d, m / m.sum(axis=(1, 2), keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-root", required=True); ap.add_argument("--frame-root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="native,yak,dino,dinol,qwen"); ap.add_argument("--views", default="agent,wrist")
    ap.add_argument("--vmax", type=float, default=4.0); ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--diff-t0", type=float, default=0.25, help="relative mode: differences below this stay gray")
    ap.add_argument("--diff", choices=["relative", "absolute"], default="relative",
                    help="relative: difference of the two displayed intensities (each surplus clipped at its own 99th "
                         "percentile and raised to GAMMA), scale +-1, gray below --diff-t0; absolute: raw surplus difference on +-VMAX")
    args = ap.parse_args()
    fams = [f for f in args.families.split(",") if os.path.exists(os.path.join(args.frame_root, f, args.tag + ".npz"))]
    missing = [f for f in args.families.split(",") if f not in fams]
    if missing:
        print("missing families (left out):", missing)
    mean = {}
    for f in fams:
        tags = sorted(x[:-4] for x in os.listdir(os.path.join(args.mean_root, f)) if x.endswith(".npz"))
        mean[f] = np.mean([layer_mean(os.path.join(args.mean_root, f, t + ".npz"))[1] for t in tags], axis=0)
    frame = {f: layer_mean(os.path.join(args.frame_root, f, args.tag + ".npz")) for f in fams}

    def surplus(f):
        m = frame[f][1]; return (m - mean[f]) * m[0].size

    os.makedirs(args.out, exist_ok=True)
    d0 = frame[fams[0]][0]
    views = [v for v in args.views.split(",")]
    rows = []
    for v, vn in enumerate(views):
        img = np.asarray(d0[vn]); big = np.asarray(Image.fromarray(img).resize((SZ, SZ), Image.BICUBIC))
        Image.fromarray(big).save(f"{args.out}/{args.prefix}_{vn}_image.png"); cells = [big]
        for f in fams:
            e = np.maximum(surplus(f)[v], 0); e = np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1) ** args.gamma
            c = (overlay(big, e, 0.55, clip_pct=100.0) * 255).round().astype(np.uint8)
            Image.fromarray(c).save(f"{args.out}/{args.prefix}_{vn}_{f}.png"); cells.append(c)
            print(f"{vn} {f}: surplus max {surplus(f)[v].max():.2f}")
        if "yak" in fams and "native" in fams:
            if args.diff == "relative":
                def shown(f):
                    e = np.maximum(surplus(f)[v], 0); return np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1) ** args.gamma
                d = shown("yak") - shown("native"); vmax, t0 = 1.0, args.diff_t0
            else:
                d = surplus("yak")[v] - surplus("native")[v]; vmax, t0 = args.vmax, T0
            heat = np.asarray(Image.fromarray(d.astype(np.float32)).resize((SZ, SZ), Image.BICUBIC))
            gray = np.asarray(Image.fromarray(big).convert("L")).astype(np.float32) / 255.0
            base = 0.75 * np.stack([gray] * 3, -1) + 0.15
            col = cm.get_cmap("RdBu_r")(np.clip((heat + vmax) / (2 * vmax), 0, 1))[..., :3]
            a = 0.85 * np.clip((np.abs(heat) - t0) / (vmax - t0), 0, 1) ** 0.6
            c = (np.clip((1 - a[..., None]) * base + a[..., None] * col, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(c).save(f"{args.out}/{args.prefix}_{vn}_diff.png"); cells.append(c)
            print(f"{vn} diff: min {d.min():.2f} max {d.max():.2f}")
        rows.append(np.concatenate(cells, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(f"{args.out}/{args.prefix}_montage.png")
    print("cells written to", args.out, "| montage columns: image |", " | ".join(fams), "| diff")


if __name__ == "__main__":
    main()
