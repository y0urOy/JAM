"""Figure-4-style readout video along one episode.

Each video frame shows, for one observation: Current frame | Recorded future (+16 steps) | Recorded change |
ImageWAM | JAM, offset [0] | JAM, offset [1].  The last four panels are signed mean-luminance changes on one
RdBu_r scale (red brighter, blue darker): the recorded future minus the current frame, and each policy's future
readout minus its current readout (same frozen features, two MLP readouts).  The colour scale is fixed for the
whole episode at the 99.5th percentile of |recorded change| over the agent view, the same rule as the paper's
panels but over all frames instead of one.

Text uses the website's typeface (Inter, --font-dir) and everything is drawn at --scale times the base layout so
the labels stay sharp on high-density screens.  The web page supplies the progress bar, so none is burned in.

usage: render_readout_video.py --native NPZ --off0 NPZ --jam NPZ --out DIR [--fps 10] [--view agent|both]
       [--scale 1.5] [--attenuate off0:0.5]
"""
import argparse, json, os, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
import matplotlib

INK = (0x29, 0x33, 0x40); MUTED = (0x7A, 0x85, 0x90); TEAL = (0x1F, 0x7A, 0x73)
FONT_DIR = "/path/to/workspace/JAM/tools/fonts"


def font(size, weight="Regular", font_dir=FONT_DIR):
    path = os.path.join(font_dir, f"Inter-{weight}.ttf")
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    fallback = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if weight == "SemiBold" else "DejaVuSans.ttf")
    return ImageFont.truetype(fallback, size)


def load(path):
    d = np.load(path, allow_pickle=True)
    # the dataset stores the prompt template around the instruction; keep only the instruction itself
    instr = re.sub(r"^A video recorded from a robot's point of view executing the following instruction:\s*", "", str(d["instruction"]), flags=re.I).strip()
    return {k: d[k] for k in ("frame_index", "current", "future", "pred_current", "pred_future")} | {"instruction": instr}


def signed(a, b):  # uint8 [N,H,W,3] -> [N,H,W] mean-luminance change in [-1,1]
    return (b.astype(np.float32) - a.astype(np.float32)).mean(-1) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True); ap.add_argument("--off0", required=True); ap.add_argument("--jam", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--view", default="agent", choices=["agent", "both"]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.5, help="drawing scale over the base 224-px panel layout")
    ap.add_argument("--font-dir", default=FONT_DIR)
    ap.add_argument("--ffmpeg", default="/path/to/workspace/miniconda3/envs/jam_multimodal-ovis-py311/bin/ffmpeg")
    ap.add_argument("--attenuate", default="", help="comma list policy:factor; positive (brightening) readout values of that "
                    "policy are multiplied by the factor before painting, e.g. off0:0.5 (the paper's Figure 4 applies 0.5 "
                    "to the JAM [0] panel)")
    args = ap.parse_args()
    S = args.scale
    px = lambda v: int(round(v * S))
    atten = {k: float(v) for k, v in (kv.split(":") for kv in args.attenuate.split(",") if kv.strip())}
    runs = {"native": load(args.native), "off0": load(args.off0), "jam": load(args.jam)}
    common = sorted(set(runs["native"]["frame_index"]) & set(runs["off0"]["frame_index"]) & set(runs["jam"]["frame_index"]))
    if args.limit: common = common[:args.limit]
    pos = {k: {int(f): i for i, f in enumerate(v["frame_index"])} for k, v in runs.items()}
    ref = runs["jam"]
    W_img = ref["current"].shape[2]; half = W_img // 2
    cols = slice(0, half) if args.view == "agent" else slice(0, W_img)
    # fixed colour scale over the episode (agent view), 99.5th percentile of |recorded change|
    true_all = signed(ref["current"], ref["future"])[:, :, :half]
    vmax = float(np.percentile(np.abs(true_all), 99.5)) + 1e-6
    print(f"{len(common)} frames, vmax {vmax:.3f}, scale {S}")
    cmap = cm.get_cmap("RdBu_r")
    CELL, GAP = px(224), px(10)
    cw = CELL if args.view == "agent" else 2 * CELL

    def paint(x):  # scalar map -> RGB at panel size (resize the field first, then colour it)
        big = np.asarray(Image.fromarray(x.astype(np.float32), mode="F").resize((cw, CELL), Image.BICUBIC))
        return (cmap(np.clip((big + vmax) / (2 * vmax), 0, 1))[..., :3] * 255).round().astype(np.uint8)

    def photo(img):
        return np.asarray(Image.fromarray(img).resize((cw, CELL), Image.LANCZOS))

    titles = ["Current frame", "Recorded future", "Recorded change", "ImageWAM", "JAM, offset [0]", "JAM, offset [1]"]
    LEFT, TOP = px(12), px(46)
    W = LEFT + 6 * (cw + GAP) - GAP + LEFT
    H = TOP + CELL + px(44)
    W += W % 2; H += H % 2          # even size, so the encoder pad never adds a black edge
    f_title, f_body, f_small = font(px(16.5), "SemiBold", args.font_dir), font(px(14.5), "Regular", args.font_dir), font(px(13), "Regular", args.font_dir)
    os.makedirs(args.out, exist_ok=True); fdir = os.path.join(args.out, "frames"); os.makedirs(fdir, exist_ok=True)
    bar_w, bar_h = px(200), px(9)
    bar = (cmap(np.linspace(0, 1, bar_w))[None, :, :3].repeat(bar_h, 0) * 255).round().astype(np.uint8)
    y_foot = TOP + CELL + px(22)
    for i, f in enumerate(common):
        j = pos["jam"][f]
        cur = ref["current"][j][:, cols]; fut = ref["future"][j][:, cols]
        true = signed(ref["current"][j:j + 1], ref["future"][j:j + 1])[0][:, cols]
        panels = [photo(cur), photo(fut), paint(true)]
        for k in ("native", "off0", "jam"):
            r = runs[k]; jj = pos[k][f]
            x = signed(r["pred_current"][jj:jj + 1], r["pred_future"][jj:jj + 1])[0][:, cols]
            if k in atten:
                x = np.where(x > 0, x * atten[k], x)
            panels.append(paint(x))
        can = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(can)
        x = LEFT
        for title, p in zip(titles, panels):
            dr.text((x + cw // 2, TOP - px(17)), title, fill=TEAL if title == "JAM, offset [1]" else INK, font=f_title, anchor="mm")
            can.paste(Image.fromarray(p), (x, TOP)); x += cw + GAP
        # footer: instruction, colour bar, step counter
        dr.text((LEFT, y_foot), f"“{ref['instruction']}”", fill=INK, font=f_body, anchor="lm")
        bx = (W - bar_w) // 2
        dr.text((bx - px(8), y_foot), "Darker", fill=MUTED, font=f_small, anchor="rm")
        can.paste(Image.fromarray(bar), (bx, y_foot - bar_h // 2))
        dr.text((bx + bar_w + px(8), y_foot), "Brighter", fill=MUTED, font=f_small, anchor="lm")
        dr.text((W - LEFT, y_foot), f"step {int(f)}  ·  +16 ahead", fill=MUTED, font=f_body, anchor="rm")
        can.save(os.path.join(fdir, f"{i:05d}.png"))
        if i % 50 == 0:
            print(f"frame {i + 1}/{len(common)}")
    json.dump({"frames": [int(f) for f in common], "vmax": vmax, "instruction": ref["instruction"], "size": [W, H],
               "attenuate": atten, "scale": S, "fps": args.fps}, open(os.path.join(args.out, "frames.json"), "w"), indent=1)
    mp4 = os.path.join(args.out, "readout_episode.mp4")
    subprocess.run([args.ffmpeg, "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i", os.path.join(fdir, "%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-movflags", "+faststart", mp4], check=True)
    print("wrote", mp4, "frames", len(common), "size", W, "x", H)


if __name__ == "__main__":
    main()
