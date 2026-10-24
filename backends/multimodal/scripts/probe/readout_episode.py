#!/usr/bin/env python3
"""Decode both readouts of a finished future-decodability probe on every frame of ONE held-out episode.

Same features and readouts as readout_diff.py (frozen policy, current-observation tokens at the probe layer,
per-token MLP readouts of the current frame and of the frame 16 steps ahead), but instead of 2,000 random
held-out windows it walks one episode in order, so the anticipated change can be shown as a video.

Writes <out>/episode_<E>.npz with uint8 arrays [N, H, W, 3]: current, future, pred_current, pred_future, plus
frame_index [N], the instruction, and the grid/patch/layer used.  Frames whose 16-step future is padded (the
end of the episode) are skipped.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import future_probe as FP  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--family", required=True, choices=["yak", "qwen"])
    ap.add_argument("--probe-out", required=True, help="finished probe directory holding readouts.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", default=None, help="probe layer to decode (default yak 11, qwen 13)")
    ap.add_argument("--dataset-index", type=int, default=3, help="position of the suite in dataset_dirs (3 = libero_10)")
    ap.add_argument("--episode", type=int, default=139, help="LeRobot episode index inside that suite")
    ap.add_argument("--anchor", default="9993:384", help="fallback held-out index:frame pair known to lie in the episode")
    ap.add_argument("--episode-length", type=int, default=450)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N frames")
    ap.add_argument("--val-proportion", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe-hidden", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def episode_range(ds, dataset_index, episode, anchor, length):
    """[from, to) in the held-out dataset's index space for the episode, verified on the dataset itself."""
    cand = []
    try:
        multi = ds.lerobot_dataset.multi_dataset
        subs = list(multi._datasets)
        offset = sum(int(d.num_frames) for d in subs[:dataset_index])
        sub = subs[dataset_index]
        eps = getattr(sub, "episodes", None)
        eps = list(eps) if eps is not None else list(range(int(sub.num_episodes)))
        pos = eps.index(episode)
        edi = sub.episode_data_index
        cand.append(("episode_data_index", offset + int(edi["from"][pos]), offset + int(edi["to"][pos])))
    except Exception as e:  # noqa: BLE001
        print(f"[episode] episode_data_index route failed: {e}", flush=True)
    if anchor:
        ai, af = (int(v) for v in anchor.split(":"))
        cand.append(("anchor", ai - af, ai - af + length))
    for name, fr, to in cand:
        ok = True
        for idx, want in ((fr, 0), (to - 1, to - fr - 1)):
            s = ds[idx]
            e, f = int(s["episode_index"]), int(s["frame_index"])
            if e != episode or f != want:
                print(f"[episode] {name}: index {idx} -> episode {e} frame {f}, expected {episode}/{want}", flush=True)
                ok = False
                break
        if ok:
            print(f"[episode] {name}: indices [{fr}, {to}) = {to - fr} frames of episode {episode}", flush=True)
            return fr, to
    raise RuntimeError("could not locate the episode in the held-out dataset")


def main():
    args = parse_args()
    run_dir = Path(args.run).resolve()
    probe_out = Path(args.probe_out).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    layer = args.layer or ("11" if args.family == "yak" else "13")
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "weights" / "step_021700.pt"
    torch.manual_seed(args.seed)
    print(f"[episode] run={run_dir} ckpt={ckpt} family={args.family} layer={layer} episode={args.episode}", flush=True)
    t0 = time.time()
    cfg, model, processor, dtype = FP.AV.build(run_dir, ckpt, device)
    model.eval()
    print(f"[episode] model loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    FP.misc.register_work_dir(str(out))
    ds = FP.build_dataset(cfg, run_dir, heldout=True, val_proportion=args.val_proportion)
    fr, to = episode_range(ds, args.dataset_index, args.episode, args.anchor, args.episode_length)
    indices = list(range(fr, to, args.stride))
    if args.limit:
        indices = indices[:args.limit]
    loader = torch.utils.data.DataLoader(FP.Indexed(ds, indices), batch_size=args.batch_size, num_workers=args.workers,
                                         shuffle=False, collate_fn=FP.collate, pin_memory=False, persistent_workers=False)
    ext = FP.Extractor(model, args.family, [layer], device)
    sd = torch.load(probe_out / "readouts.pt", map_location="cpu")
    readouts = None
    cur_all, fut_all, pc_all, pf_all, fidx = [], [], [], [], []
    instruction = None
    t0 = time.time()
    for bi, batch in enumerate(loader):
        video = FP.video_to_float(batch["video"])
        pad = batch["image_is_pad"]
        keep = ~(pad.reshape(pad.shape[0], -1).any(dim=1))
        if not bool(keep.any()):
            continue
        ki = keep.nonzero(as_tuple=True)[0]
        cur, fut = video[ki, :, 0], video[ki, :, -1]
        instr = [batch["instruction"][int(i)] for i in ki]
        proprio = batch["proprio"][ki]
        feats, grid, patch = ext(cur, instr, proprio)
        f = feats[layer]
        if readouts is None:
            d_in = int(f.shape[-1])
            readouts = {}
            for t in ("future", "current"):
                r = FP.PatchReadout(d_in, args.probe_hidden, patch).to(device)
                r.load_state_dict(sd[f"{layer}|{t}"]); r.eval()
                readouts[t] = r
            instruction = instr[0]
        pf = FP.predict_images(readouts["future"], f.numpy(), grid, patch, device)
        pc = FP.predict_images(readouts["current"], f.numpy(), grid, patch, device)
        cur_all.append(FP.to_u8(cur)); fut_all.append(FP.to_u8(fut)); pf_all.append(pf); pc_all.append(pc)
        fidx.extend(int(batch["frame_index"][int(i)]) for i in ki)
        if bi % 10 == 0:
            print(f"[episode] {len(fidx)} frames  {(time.time() - t0) / 60:.1f} min", flush=True)
    if not fidx:
        raise RuntimeError("no frames extracted")
    arr = {k: np.concatenate(v, 0) for k, v in (("current", cur_all), ("future", fut_all), ("pred_current", pc_all), ("pred_future", pf_all))}
    path = out / f"episode_{args.episode}.npz"
    np.savez_compressed(path, frame_index=np.asarray(fidx, dtype=np.int32), instruction=instruction, episode=args.episode,
                        dataset_index=args.dataset_index, grid=np.asarray(grid), patch=patch, layer=layer, run=str(run_dir), **arr)
    (out / "episode_meta.json").write_text(json.dumps({"episode": args.episode, "frames": len(fidx), "first": fidx[0], "last": fidx[-1],
                                                       "layer": layer, "run": str(run_dir), "probe_out": str(probe_out)}, indent=1))
    print(f"READOUT_EPISODE_DONE {path} frames={len(fidx)} first={fidx[0]} last={fidx[-1]}", flush=True)


if __name__ == "__main__":
    main()
