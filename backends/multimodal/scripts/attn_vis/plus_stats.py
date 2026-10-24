"""Aggregate attention statistics over proxy-perturbed frames.

Usage: python plus_stats.py --root OUTROOT --order yak,dino,dinol,videomae,qwen --out plus_stats.json
For every family and record: ROI density ratios (record-specific boxes), normalized entropy of the
agent-view map, visual share of action attention, agent-view share.  Aggregates per category and
reports the change relative to the clean frame of the same task.
"""
import argparse, glob, json, os
import numpy as np
import make_figure as mf
from make_figure import pick
from roi_stats import roi_mass

KEY_OBJ = {"libero_goal": ["bowl", "stove", "gripper"], "libero_10": ["white_mug", "plate", "gripper"]}
NAMES = {"yak": "Yak18", "dino": "DINOv2-B", "dinol": "DINOv2-L", "videomae": "VideoMAE-B", "qwen": "Qwen3.5-2B"}


def entropy(m):
    p = m / m.sum(); return float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--order", default="yak,dino,dinol,videomae,qwen"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); per = {}
    for fam in a.order.split(","):
        for f in sorted(glob.glob(os.path.join(a.root, fam, "*.npz"))):
            d = np.load(f, allow_pickle=True); rec = json.loads(str(d["record"]))
            m = pick(d["maps"], d["layers"], "mean"); keep = np.asarray(d["layers"]) < 100
            vis_share = float(d["maps"][keep].mean(axis=1).sum(axis=(1, 2, 3)).mean())
            task = "libero_goal" if "libero_goal" in rec["tag"] else "libero_10"
            row = {"category": rec["category"], "strength": rec["strength"], "task": task,
                   "entropy_agent": entropy(m[0]), "visual_share": vis_share, "agent_share": float(m[0].sum() / m.sum())}
            for k in KEY_OBJ[task]:
                mass, area = roi_mass(m[0], rec["rois"][k]); row[f"ratio_{k}"] = mass / max(area, 1e-9)
            per.setdefault(fam, []).append(row)
    # aggregate
    summary = {}
    for fam, rows in per.items():
        summary[fam] = {}
        for task in ("libero_goal", "libero_10"):
            clean = next(r for r in rows if r["task"] == task and r["category"] == "clean")
            obj = KEY_OBJ[task][0]; tgt = KEY_OBJ[task][1]
            for cat in ("camera", "lighting", "noise"):
                rs = [r for r in rows if r["task"] == task and r["category"] == cat]
                summary[fam][f"{task}/{cat}"] = {
                    "n": len(rs),
                    f"{obj}_ratio_clean": round(clean[f"ratio_{obj}"], 2),
                    f"{obj}_ratio_mean": round(float(np.mean([r[f"ratio_{obj}"] for r in rs])), 2),
                    f"{obj}_retained": round(float(np.mean([r[f"ratio_{obj}"] for r in rs]) / max(clean[f"ratio_{obj}"], 1e-9)), 2),
                    f"{tgt}_ratio_clean": round(clean[f"ratio_{tgt}"], 2),
                    f"{tgt}_ratio_mean": round(float(np.mean([r[f"ratio_{tgt}"] for r in rs])), 2),
                    "entropy_clean": round(clean["entropy_agent"], 3),
                    "entropy_mean": round(float(np.mean([r["entropy_agent"] for r in rs])), 3),
                    "visual_share_clean": round(clean["visual_share"], 3),
                    "visual_share_mean": round(float(np.mean([r["visual_share"] for r in rs])), 3),
                }
    json.dump({"per_record": per, "summary": summary}, open(a.out, "w"), indent=1)
    # console table: retained object density per category (mean over the two tasks)
    print(f"{'backbone':12s} " + " ".join(f"{c:>18s}" for c in ("camera", "lighting", "noise")) + "   (retained object density = perturbed/clean, mean of both tasks; entropy change)")
    for fam in a.order.split(","):
        cells = []
        for cat in ("camera", "lighting", "noise"):
            vals = []; dent = []
            for task in ("libero_goal", "libero_10"):
                s = summary[fam][f"{task}/{cat}"]; obj = KEY_OBJ[task][0]
                vals.append(s[f"{obj}_retained"]); dent.append(s["entropy_mean"] - s["entropy_clean"])
            cells.append(f"{np.mean(vals):.2f} (dH {np.mean(dent):+.3f})")
        print(f"{NAMES.get(fam, fam):12s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
