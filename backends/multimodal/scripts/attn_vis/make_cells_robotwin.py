"""Figure-5-style head-camera cells for the RoboTwin captures of attn_vis_robotwin.py.

Families: flux2_native (released ImageWAM FLUX.2), flux2_jam (JAM FLUX.2), qwen_rt (JAM Qwen3.5-2B).  The FLUX.2
map covers the compact 288x256 layout as one 18x16 grid whose top 12 rows are the head camera; the Qwen map has
three 7x7 views with the head camera first.  The head-camera map of each frame is normalized within the head
region, the family's mean over the rand* frames of the same capture is subtracted, the surplus is clipped at
zero, scaled to its 99th percentile, raised to GAMMA and jet-overlaid on the 4:3 head image (448x336).  The
difference cell is surplus(flux2_jam) - surplus(flux2_native) on a fixed +-VMAX scale, grey below 0.5.
usage: make_cells_robotwin.py --root DIR --tag TAG --prefix P --out DIR [--vmax 4] [--gamma 0.6]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from matplotlib import cm
from make_figure import overlay

W, H = 448, 336
T0 = 0.5
FAMS = ["flux2_native", "flux2_jam", "qwen_rt"]   # --native/--jam rename the pair (e.g. wan_native / wan_foresight)
SQUARE = False   # --square: centre-crop the 4:3 head camera to a square (FLUX.2 grid 12x16 -> 12x12, Qwen 7x7 -> centre 7x5.25 kept as 7x7 by cropping the image only)


LAYERS = None   # --layers a:b restricts the layer mean to captured layers a..b-1 (both policies alike)


def head_map(path):
    d = np.load(path, allow_pickle=True)
    maps = d["maps"] if LAYERS is None else d["maps"][LAYERS[0]:LAYERS[1]]
    m = maps.mean(axis=(0, 1))                                 # [V, gh, gw]
    fam = str(d["family"])
    if fam == "flux2":
        if m.shape != (1, 18, 16):
            raise ValueError(f"unexpected FLUX.2 grid {m.shape}")
        h = m[0, :12, :]                                         # head camera rows of the compact layout
        if SQUARE:
            h = h[:, 2:14]                                       # centre 12 of 16 columns
    elif fam == "wan":
        if m.shape != (1, 12, 10):
            raise ValueError(f"unexpected Wan grid {m.shape}")
        h = m[0, :8, :]                                          # head camera = top 256 of 384 rows (8 of 12)
        if SQUARE:
            h = h[:, 1:9]                                        # centre 8 of 10 columns
    else:
        if m.shape[0] != 3 or m.shape[1:] != (7, 7):
            raise ValueError(f"unexpected Qwen grid {m.shape}")
        h = m[0]
        if SQUARE:                                               # 7 columns cover 4:3; keep the centre 5.25 by interpolation
            xs = np.linspace(0.875, 6.125, 7)
            cols = np.arange(7); h = np.stack([np.interp(xs, cols, row) for row in h])
    return d, h / h.sum()


def crop_head(img):
    if not SQUARE:
        return img
    hgt, wid = img.shape[:2]; side = min(hgt, wid); x0 = (wid - side) // 2
    return img[:, x0:x0 + side]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--vmax", type=float, default=4.0); ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--jam", default="flux2_jam", help="capture family written as the flux2_jam cell (e.g. flux2_foresight)")
    ap.add_argument("--native", default="flux2_native", help="capture family written as the flux2_native cell (e.g. wan_native)")
    ap.add_argument("--square", action="store_true"); ap.add_argument("--mean-root", default=None, help="capture dir whose rand* frames give the mean map (default: --root)")
    ap.add_argument("--layers", default=None, help="a:b layer band for the layer mean (default all captured layers)")
    args = ap.parse_args()
    global SQUARE, W, H, LAYERS
    SQUARE = args.square
    if args.layers: LAYERS = tuple(int(x) for x in args.layers.split(":"))
    if SQUARE: W = H = 448
    FAMS[0] = args.native; FAMS[1] = args.jam
    fams = [f for f in FAMS if os.path.exists(os.path.join(args.root, f, args.tag + ".npz"))]
    print("families:", fams)
    mean, frame = {}, {}
    mroot = args.mean_root or args.root
    for f in fams:
        rands = sorted(x[:-4] for x in os.listdir(os.path.join(mroot, f)) if x.startswith("rand") and x.endswith(".npz"))
        mean[f] = np.mean([head_map(os.path.join(mroot, f, t + ".npz"))[1] for t in rands], axis=0)
        frame[f] = head_map(os.path.join(args.root, f, args.tag + ".npz"))
        print(f, "mean over", len(rands), "frames")

    def surplus(f):
        m = frame[f][1]; return (m - mean[f]) * m.size

    os.makedirs(args.out, exist_ok=True)
    head = crop_head(np.asarray(frame[fams[0]][0]["head"]))
    big = np.asarray(Image.fromarray(head).resize((W, H), Image.BICUBIC))
    Image.fromarray(big).save(f"{args.out}/{args.prefix}_head_image.png"); cells = [big]
    for f in fams:
        e = np.maximum(surplus(f), 0); e = np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1) ** args.gamma
        c = (overlay(big, e, 0.55, clip_pct=100.0) * 255).round().astype(np.uint8)
        name = "flux2_jam" if f == args.jam else ("flux2_native" if f == args.native else f)
        Image.fromarray(c).save(f"{args.out}/{args.prefix}_head_{name}.png"); cells.append(c)
        print(f"{f}: surplus max {surplus(f).max():.2f}, visual share {np.nanmean(frame[f][0]['group_mass'][..., 1]):.3f}")
    if args.jam in fams and args.native in fams:
        # difference of the displayed intensities (each surplus scaled by its own 99th percentile and
        # compressed), fixed +-1 scale, gray below 0.25: the definition of Figures 5 and 8
        def shown(f):
            e = np.maximum(surplus(f), 0); return np.clip(e / (np.percentile(e, 99) + 1e-9), 0, 1) ** args.gamma
        d = shown(args.jam) - shown(args.native); vmax, t0 = 1.0, 0.25
        heat = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W, H), Image.BICUBIC))
        gray = np.asarray(Image.fromarray(big).convert("L")).astype(np.float32) / 255.0
        base = 0.75 * np.stack([gray] * 3, -1) + 0.15
        col = cm.get_cmap("RdBu_r")(np.clip((heat + vmax) / (2 * vmax), 0, 1))[..., :3]
        a = 0.85 * np.clip((np.abs(heat) - t0) / (vmax - t0), 0, 1) ** 0.6
        c = (np.clip((1 - a[..., None]) * base + a[..., None] * col, 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(c).save(f"{args.out}/{args.prefix}_head_diff.png"); cells.append(c)
        print(f"diff: min {d.min():.2f} max {d.max():.2f}")
    Image.fromarray(np.concatenate(cells, axis=1)).save(f"{args.out}/{args.prefix}_montage.png")
    print("cells written to", args.out)


if __name__ == "__main__":
    main()
