"""Decode RoboTwin 2.0 demonstration frames (three cameras + 14-D state) into an attention-capture manifest.

Records come from the LeRobot v2.1 dataset used for training (50 tasks x 550 episodes: the first 50 of each
block are the Clean demos, the other 500 the Randomized ones).  Each record decodes cam_high, cam_left_wrist and
cam_right_wrist at one frame with PyAV, reads observation.state at that frame from the parquet file, and takes the
episode's first instruction.  Usage:
  python make_robotwin_manifest.py --out DIR --record TAG:EPISODE:FRAME ... [--random N --seed S]
"""
import argparse, json, random
from pathlib import Path
import av, numpy as np, pyarrow.parquet as pq
from PIL import Image

ROOT = Path("./data/robotwin2.0/robotwin2.0")
CAMS = {"head": "observation.images.cam_high", "left": "observation.images.cam_left_wrist", "right": "observation.images.cam_right_wrist"}


def decode(ep: int, frame: int, cam_key: str) -> np.ndarray:
    path = ROOT / f"videos/chunk-{ep // 1000:03d}/{cam_key}/episode_{ep:06d}.mp4"
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, fr in enumerate(container.decode(stream)):
            if i == frame:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(f"frame {frame} not in {path}")


def episode_meta():
    rows = {}
    for line in open(ROOT / "meta/episodes.jsonl"):
        r = json.loads(line); rows[int(r["episode_index"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", action="append", default=[], help="TAG:EPISODE:FRAME or TAG:EPISODE:FRAC (0<frac<1)")
    ap.add_argument("--random", type=int, default=0, help="add N random (episode, frame) records for the mean map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-all-tasks", action="store_true", help="add one Clean and one Randomized episode of every task")
    ap.add_argument("--clean-ep", default="3", help="comma-separated episodes within the task block for the Clean sweep (0-49)")
    ap.add_argument("--rand-ep", default="120", help="comma-separated episodes within the task block for the Randomized sweep (50-549)")
    ap.add_argument("--ahead2", type=int, default=0, help="second, longer look-ahead of the head camera (clipped)")
    ap.add_argument("--phases", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--ahead", type=int, default=0, help="also decode the head camera AHEAD frames later (clipped) for change maps")
    args = ap.parse_args()
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    meta = episode_meta()
    specs = []
    for r in args.record:
        tag, ep, fr = r.split(":"); ep = int(ep)
        length = int(meta[ep]["length"])
        frame = int(round(float(fr) * (length - 1))) if "." in fr else int(fr)
        specs.append((tag, ep, frame))
    if args.sweep_all_tasks:
        for b in range(50):
            for kind, eps in (("C", args.clean_ep), ("R", args.rand_ep)):
                for e in (int(x) for x in eps.split(",")):
                    ep = b * 550 + e; length = int(meta[ep]["length"])
                    for ph in args.phases.split(","):
                        specs.append((f"t{b:02d}{kind}_e{ep}_p{int(round(float(ph)*100)):02d}", ep, int(round(float(ph) * (length - 1)))))
    rng = random.Random(args.seed)
    for i in range(args.random):
        ep = rng.randrange(len(meta)); length = int(meta[ep]["length"])
        specs.append((f"rand{i:02d}_e{ep}", ep, rng.randrange(int(0.05 * length), int(0.95 * length))))
    records = []
    for tag, ep, frame in specs:
        table = pq.read_table(ROOT / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet", columns=["observation.state", "frame_index"])
        states = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
        rec = {"tag": tag, "episode_index": ep, "frame_index": int(frame), "episode_length": int(meta[ep]["length"]),
               "instruction": meta[ep]["tasks"][0], "state": [float(v) for v in states[frame]]}
        for name, key in CAMS.items():
            p = out / "frames" / f"{tag}_f{frame}_{name}.png"
            if p.exists():
                img = np.asarray(Image.open(p))          # reuse a frame decoded by an interrupted run
            else:
                img = decode(ep, frame, key); Image.fromarray(img).save(p)
            rec[name] = str(p); rec[f"{name}_shape"] = list(img.shape)
        if args.ahead > 0:
            ff = min(int(meta[ep]["length"]) - 1, frame + args.ahead)
            p = out / "frames" / f"{tag}_f{ff}_head_future.png"
            if not p.exists():
                img = decode(ep, ff, CAMS["head"]); Image.fromarray(img).save(p)
            rec["future_frame_index"] = ff; rec["future_head"] = str(p)
        if args.ahead2 > 0:
            ff = min(int(meta[ep]["length"]) - 1, frame + args.ahead2)
            p = out / "frames" / f"{tag}_f{ff}_head_future2.png"
            if not p.exists():
                img = decode(ep, ff, CAMS["head"]); Image.fromarray(img).save(p)
            rec["future2_frame_index"] = ff; rec["future2_head"] = str(p)
        records.append(rec); print(tag, ep, frame, rec["instruction"][:70], states[frame].round(3)[:4])
    json.dump(records, open(out / "manifest.json", "w"), indent=1)
    print("wrote", out / "manifest.json", len(records), "records")


if __name__ == "__main__":
    main()
