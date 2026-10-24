"""Figure-5-style attention video along one episode.

Each video frame is the Figure 5 grid for one observation of the episode: Image | ImageWAM | JAM | Difference |
JAM (DINOv2-L) | JAM (Qwen3.5-2B), rows = agent and wrist views, with the policy's text/image/action attention
shares beside its maps.  The display transform follows make_cells_excess_ext.py (surplus of the layer-mean map over
the policy's 100-frame mean map, clipped at zero, compressive power GAMMA, jet overlay), except that the surplus
scale is fixed per policy and view over the whole episode (its 99th percentile) instead of per frame, so the
brightness does not flicker between frames.  The difference cell is shown(JAM) - shown(ImageWAM) on a +-1 scale.

Text uses the website's typeface (Inter, --font-dir) and everything is drawn at --scale times the base layout so
the labels stay sharp on high-density screens.  The web page supplies the progress bar, so none is burned in.

usage: render_attention_video.py --mean-root DIR --frame-root DIR --out DIR [--fps 10] [--gamma 0.6] [--scale 1.5]
       [--families native,yak,dinol,qwen] [--limit N] [--ffmpeg PATH]
"""
import argparse, json, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import matplotlib
from make_figure import overlay

COLS = [("native", "ImageWAM"), ("yak", "JAM"), ("diff", "Difference"), ("dinol", "JAM (DINOv2-L)"), ("qwen", "JAM (Qwen3.5-2B)")]
# Figure 5 palette (Figures 1/2 deck colours, 50 % tints) for text / image / action keys
SHARE_COL = {"text": ("#AD7158", "#D6B8AB"), "visual": ("#626B98", "#B1B5CC"), "action": ("#427F7A", "#A1BFBD")}
INK = (0x29, 0x33, 0x40); MUTED = (0x7A, 0x85, 0x90)
FONT_DIR = "/path/to/workspace/JAM/tools/fonts"


def font(size, weight="Regular", font_dir=FONT_DIR):
    path = os.path.join(font_dir, f"Inter-{weight}.ttf")
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    fallback = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if weight == "SemiBold" else "DejaVuSans.ttf")
    return ImageFont.truetype(fallback, size)


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def layer_mean(path):
    d = np.load(path, allow_pickle=True)
    keep = np.asarray(d["layers"]) < 100
    m = d["maps"][keep].mean(axis=(0, 1))                       # [V, gh, gw]
    return d, m / m.sum(axis=(1, 2), keepdims=True)


def shares(d):
    g = d["group_mass"]                                           # [L, S, 4] text, visual, action, other
    keep = np.asarray(d["layers"]) < 100
    v = np.nanmean(g[keep][..., :3], axis=(0, 1))
    return v / max(v.sum(), 1e-9)


def frame_num(tag):
    m = re.search(r"_f(\d+)$", tag)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-root", required=True); ap.add_argument("--frame-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--families", default="native,yak,dinol,qwen"); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.6); ap.add_argument("--diff-t0", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.5, help="drawing scale over the base 224-px cell layout")
    ap.add_argument("--font-dir", default=FONT_DIR)
    ap.add_argument("--ffmpeg", default="/path/to/workspace/miniconda3/envs/jam_multimodal-ovis-py311/bin/ffmpeg")
    ap.add_argument("--title", default=None, help="footer text; default = the instruction from the capture")
    args = ap.parse_args()
    S = args.scale
    px = lambda v: int(round(v * S))
    fams = [f for f in args.families.split(",") if os.path.isdir(os.path.join(args.frame_root, f))]
    tags = None
    for f in fams:
        t = {x[:-4] for x in os.listdir(os.path.join(args.frame_root, f)) if x.endswith(".npz")}
        tags = t if tags is None else tags & t
    tags = sorted(tags, key=frame_num)[::args.stride]
    if args.limit: tags = tags[:args.limit]
    print(f"families {fams}; {len(tags)} frames; scale {S}")

    mean = {}
    for f in fams:
        mt = sorted(x[:-4] for x in os.listdir(os.path.join(args.mean_root, f)) if x.endswith(".npz"))
        mean[f] = np.mean([layer_mean(os.path.join(args.mean_root, f, t + ".npz"))[1] for t in mt], axis=0)
    # pass 1: surplus maps, images, shares; fixed 99th-percentile scale per (family, view) over the episode
    data = {f: [] for f in fams}; imgs = []; shr = {f: [] for f in fams}; instr = None
    for t in tags:
        for f in fams:
            d, m = layer_mean(os.path.join(args.frame_root, f, t + ".npz"))
            data[f].append(np.maximum((m - mean[f]) * m[0].size, 0)); shr[f].append(shares(d))
            if f == fams[0]:
                imgs.append((np.asarray(d["agent"]), np.asarray(d["wrist"])))
                if instr is None:
                    instr = str(d["instruction"])
    scale = {f: [np.percentile(np.stack([x[v] for x in data[f]]), 99) + 1e-9 for v in range(2)] for f in fams}
    print("scales:", {f: [round(float(s), 2) for s in scale[f]] for f in fams})

    os.makedirs(args.out, exist_ok=True); fdir = os.path.join(args.out, "frames"); os.makedirs(fdir, exist_ok=True)
    CELL, GAP, BAR_W = px(224), px(10), px(22)
    f_title = font(px(17), "SemiBold", args.font_dir)
    f_row = font(px(15), "Medium", args.font_dir)
    f_num = font(px(12.5), "Medium", args.font_dir)
    f_body = font(px(15), "Regular", args.font_dir)
    LEFT, TOP = px(34), px(58)
    W = LEFT + CELL + GAP + sum(CELL + GAP + (BAR_W + GAP if c[0] != "diff" else 0) for c in COLS) - GAP + px(10)
    H = TOP + 2 * CELL + GAP + px(42)
    W += W % 2; H += H % 2          # even size, so the encoder pad never adds a black edge
    y_foot = TOP + 2 * CELL + GAP + px(21)
    cmap_diff = cm.get_cmap("RdBu_r")
    views = ("Agent view", "Wrist view")
    for i, t in enumerate(tags):
        can = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(can)
        # row labels, rotated, centred on each row
        for v, name in enumerate(views):
            lab = Image.new("RGBA", (CELL, px(26)), (255, 255, 255, 0))
            ImageDraw.Draw(lab).text((CELL // 2, px(13)), name, fill=INK, font=f_row, anchor="mm")
            rot = lab.rotate(90, expand=True)
            can.paste(rot, (px(4), TOP + v * (CELL + GAP)), rot)
        x = LEFT
        # image column
        dr.text((x + CELL // 2, TOP - px(19)), "Image", fill=INK, font=f_title, anchor="mm")
        for v in range(2):
            can.paste(Image.fromarray(imgs[i][v]).resize((CELL, CELL), Image.LANCZOS), (x, TOP + v * (CELL + GAP)))
        x += CELL + GAP
        shown = {}
        for f in fams:
            shown[f] = [np.clip(data[f][i][v] / scale[f][v], 0, 1) ** args.gamma for v in range(2)]
        for fam, title in COLS:
            dr.text((x + CELL // 2, TOP - px(19)), title, fill=INK, font=f_title, anchor="mm")
            for v in range(2):
                big = np.asarray(Image.fromarray(imgs[i][v]).resize((CELL, CELL), Image.LANCZOS))
                if fam == "diff":
                    if "yak" in shown and "native" in shown:
                        d = shown["yak"][v] - shown["native"][v]
                        heat = np.asarray(Image.fromarray(d.astype(np.float32)).resize((CELL, CELL), Image.BICUBIC))
                        gray = np.asarray(Image.fromarray(big).convert("L")).astype(np.float32) / 255.0
                        base = 0.75 * np.stack([gray] * 3, -1) + 0.15
                        col = cmap_diff(np.clip((heat + 1) / 2, 0, 1))[..., :3]
                        a = 0.85 * np.clip((np.abs(heat) - args.diff_t0) / (1 - args.diff_t0), 0, 1) ** 0.6
                        cell = (np.clip((1 - a[..., None]) * base + a[..., None] * col, 0, 1) * 255).round().astype(np.uint8)
                    else:
                        cell = np.full((CELL, CELL, 3), 235, np.uint8)
                elif fam in shown:
                    cell = (overlay(big, shown[fam][v], 0.55, clip_pct=100.0) * 255).round().astype(np.uint8)
                else:
                    cell = np.full((CELL, CELL, 3), 235, np.uint8)
                can.paste(Image.fromarray(cell), (x, TOP + v * (CELL + GAP)))
            x += CELL + GAP
            if fam != "diff":
                # stacked share bar beside the policy's maps: action (top), image, text (bottom), as in Figure 5
                if fam in shr:
                    s = shr[fam][i]; hbar = 2 * CELL + GAP; y = TOP
                    for key, val in (("action", s[2]), ("visual", s[1]), ("text", s[0])):
                        h = int(round(hbar * val)); edge, fill = SHARE_COL[key]
                        dr.rectangle([x, y, x + BAR_W, min(TOP + hbar, y + h)], fill=hexrgb(fill), outline=hexrgb(edge), width=max(1, px(1)))
                        if h >= px(18):
                            dr.text((x + BAR_W // 2 + 1, y + h // 2), f"{100 * val:.0f}", fill=INK, font=f_num, anchor="mm")
                        y += h
                x += BAR_W + GAP
        # footer: instruction and step counter (the page draws the progress bar)
        fi = frame_num(t)
        dr.text((LEFT, y_foot), args.title or f"“{instr}”", fill=INK, font=f_body, anchor="lm")
        dr.text((W - px(10), y_foot), f"step {fi}", fill=MUTED, font=f_body, anchor="rm")
        can.save(os.path.join(fdir, f"{i:05d}.png"))
        if i % 25 == 0:
            print(f"frame {i + 1}/{len(tags)} ({t})")
    json.dump({"tags": tags, "families": fams, "scales": {f: [float(s) for s in scale[f]] for f in fams},
               "instruction": instr, "size": [W, H], "scale": S, "fps": args.fps}, open(os.path.join(args.out, "frames.json"), "w"), indent=1)
    mp4 = os.path.join(args.out, "attention_episode.mp4")
    cmd = [args.ffmpeg, "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", os.path.join(fdir, "%05d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-movflags", "+faststart", mp4]
    subprocess.run(cmd, check=True)
    print("wrote", mp4, "frames", len(tags), "size", W, "x", H)


if __name__ == "__main__":
    main()
