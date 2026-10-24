"""Attention mass inside task-relevant regions (boxes in 512-px frame coordinates).

Usage: python roi_stats.py --root OUTROOT --order yak,dino,dinol,videomae,qwen [--layer mean]
Prints, per family and task, the fraction of attention mass (agent view) inside each ROI
relative to the ROI's area fraction (a ratio of 1 means uniform attention).
"""
import argparse, glob, json, os
import numpy as np
from make_figure import pick

# (x0, y0, x1, y1) in the 512x512 agent-view frame; the model sees a 224x224 resize (no crop).
ROIS = {
    "libero_goal_no_noops_lerobot_t4_e164_f32": {"bowl": (215, 245, 300, 320), "stove": (95, 190, 210, 265), "gripper": (175, 175, 300, 265)},
    "libero_10_no_noops_lerobot_t8_e309_f60": {"white_mug": (300, 195, 395, 285), "red_mug": (200, 175, 275, 275), "plate": (190, 385, 320, 485), "pudding": (150, 300, 205, 345), "gripper": (230, 120, 400, 200)},
}


def roi_mass(m, box, side=512):
    gh, gw = m.shape
    x0, y0, x1, y1 = box
    c0, c1 = int(np.floor(x0 / side * gw)), int(np.ceil(x1 / side * gw))
    r0, r1 = int(np.floor(y0 / side * gh)), int(np.ceil(y1 / side * gh))
    sub = m[r0:r1, c0:c1]
    return float(sub.sum() / m.sum()), float(sub.size / m.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen")
    ap.add_argument("--layer", default="mean"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); res = {}
    for fam in a.order.split(","):
        for f in sorted(glob.glob(os.path.join(a.root, fam, "*.npz"))):
            tag = os.path.basename(f)[:-4]; d = np.load(f, allow_pickle=True)
            m = pick(d["maps"], d["layers"], a.layer)[0]  # agent view
            row = {}
            for name, box in ROIS.get(tag, {}).items():
                mass, area = roi_mass(m, box); row[name] = {"mass": round(mass, 3), "area": round(area, 3), "ratio": round(mass / max(area, 1e-9), 2)}
            res.setdefault(fam, {})[tag] = row
            print(fam, tag, " ".join(f"{k}:{v['mass']:.2f}({v['ratio']:.1f}x)" for k, v in row.items()))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
