"""Attention-capture manifest from LIBERO-Plus evaluator observation dumps (JAM_DUMP_OBS_DIR).

Each dump folder <suite>_t<task>_ep<ep>/step_XXXX.npz holds the two rotated 256x256 camera images the policy
saw at a replan step, the raw 8-D simulator state and the task description.  Records are written in the same
format as scripts/attn_vis/manifest.json (image, wrist_image, state, instruction, suite, task_index, ...).
usage: make_plus_manifest.py --out DIR --record TAG:DUMP_DIR:STEP ...   (STEP = step index in the file name)
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--record", action="append", default=[])
    args = ap.parse_args()
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    records = []
    for spec in args.record:
        tag, dump_dir, step = spec.rsplit(":", 2)
        d = np.load(Path(dump_dir) / f"step_{int(step):04d}.npz", allow_pickle=True)
        name = Path(dump_dir).name                                   # <suite>_t<task>_ep<ep>
        suite, rest = name.split("_t", 1); task_id, ep = rest.split("_ep")
        rec = {"tag": tag, "suite": suite, "task_index": int(task_id), "episode_index": int(ep), "frame_index": int(step),
               "instruction": str(d["task"]), "state": [float(v) for v in d["state"]], "source": str(Path(dump_dir) / f"step_{int(step):04d}.npz")}
        for key, arr in (("image", d["agent"]), ("wrist_image", d["wrist"])):
            p = out / "frames" / f"{tag}_{key}.png"; Image.fromarray(np.asarray(arr)).save(p)
            rec[key] = str(p); rec[key + "_shape"] = list(np.asarray(arr).shape)
        records.append(rec); print(tag, name, step, rec["instruction"][:70])
    json.dump(records, open(out / "manifest.json", "w"), indent=1)
    print("wrote", out / "manifest.json", len(records))


if __name__ == "__main__":
    main()
