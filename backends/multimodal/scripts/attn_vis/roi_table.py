"""LaTeX table of task-relevant attention density (ratio to uniform) per backbone.

Usage: python roi_table.py --root OUTROOT --order yak,dino,dinol,videomae,qwen --out table.tex
Also reports the share of action-token attention placed on visual keys and the agent-view share.
"""
import argparse, glob, os
import numpy as np
import make_figure as mf
from make_figure import pick
from roi_stats import ROIS, roi_mass

NAMES = {"yak": "Yak18 (image DiT)", "dino": "DINOv2-B (SSL)", "dinol": "DINOv2-L (SSL)", "videomae": "VideoMAE-B (SSL)", "qwen": "Qwen3.5-2B (VLM)"}
COLS = [("libero_goal_no_noops_lerobot_t4_e164_f32", ["bowl", "stove", "gripper"]),
        ("libero_10_no_noops_lerobot_t8_e309_f60", ["white_mug", "red_mug", "plate", "gripper"])]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen"); ap.add_argument("--out", required=True)
    ap.add_argument("--include-aux", action="store_true")
    a = ap.parse_args(); rows = []
    mf.INCLUDE_AUX = bool(a.include_aux)
    for fam in a.order.split(","):
        cells = []; extra = []
        for tag, keys in COLS:
            f = os.path.join(a.root, fam, tag + ".npz")
            if not os.path.exists(f):
                cells += ["--"] * len(keys); extra += ["--", "--"]; continue
            d = np.load(f, allow_pickle=True); m = pick(d["maps"], d["layers"], "mean")
            for k in keys:
                mass, area = roi_mass(m[0], ROIS[tag][k]); cells.append(f"{mass / area:.1f}")
            keep = np.asarray(d["layers"]) < 100 if not mf.INCLUDE_AUX else np.ones(len(d["layers"]), dtype=bool)
            vm = d["maps"][keep].mean(axis=1).sum(axis=(1, 2, 3)).mean()  # visual mass, mean over joint layers
            extra.append(f"{100 * vm:.0f}"); extra.append(f"{100 * m[0].sum() / m.sum():.0f}")
        rows.append((NAMES.get(fam, fam), cells, extra))
    with open(a.out, "w") as fh:
        fh.write("\\begin{table}[htbp]\n\\caption{CAPTION}\n\\label{tab:attention}\n\\centering\n\\small\\setlength{\\tabcolsep}{4pt}\n")
        fh.write("\\begin{tabular}{@{}l ccc cccc cc@{}}\n\\toprule\n")
        fh.write(" & \\multicolumn{3}{c}{Bowl on stove} & \\multicolumn{4}{c}{Mug on plate} & \\multicolumn{2}{c}{Visual share (\\%)} \\\\\n")
        fh.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\n")
        fh.write("Backbone & bowl & stove & gripper & white mug & red mug & plate & gripper & of action attn. & on agent view \\\\\n\\midrule\n")
        for name, cells, extra in rows:
            fh.write(name + " & " + " & ".join(cells) + " & " + extra[0] + " & " + extra[1] + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
