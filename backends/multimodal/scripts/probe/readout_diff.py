#!/usr/bin/env python3
"""Anticipated-change probe: where does the future readout differ from the current readout?

Both readouts of `future_probe.py` decode the SAME current-observation features of a frozen
policy: one is trained to reconstruct the current frame, the other the frame 16 steps ahead.
Their difference is what the features say will change.  This script re-extracts the held-out
features of a finished probe run, decodes both heads, and asks whether that difference lands
on the pixels that actually move.

Reported per policy and layer, over the held-out windows:
  ratio       mean |future readout - current readout| inside the motion mask, divided by outside
  corr        Pearson correlation between |readout difference| and |future - current|
  agree       fraction of changed pixels where the signed readout difference has the sign of the
              signed frame difference
  null_*      the same three quantities against a shuffled pairing (another window's future),
              which is the chance level for this mask and decoder
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--heldout-samples", type=int, default=2000)
    ap.add_argument("--val-proportion", type=float, default=0.05)
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--probe-hidden", type=int, default=2048)
    ap.add_argument("--vis-per-suite", type=int, default=3)
    ap.add_argument("--vis-positions", default=None, help="comma list of held-out positions to render instead of the per-suite selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    # kept so that FP helpers that read args see the same fields
    ap.add_argument("--train-samples", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--probe-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--generate", type=int, default=0)
    return ap.parse_args()


def stats_one(diff_signed, g_signed, mask):
    """diff_signed, g_signed: [H,W,3] float in [-1,1] units; mask: [H,W] bool."""
    d_mag = np.abs(diff_signed).mean(-1)
    g_mag = np.abs(g_signed).mean(-1)
    inside, outside = d_mag[mask], d_mag[~mask]
    ratio = float(inside.mean() / max(outside.mean(), 1e-8)) if inside.size and outside.size else float("nan")
    corr = float(np.corrcoef(d_mag.ravel(), g_mag.ravel())[0, 1])
    sel = mask[..., None] & (np.abs(g_signed) > 0.05)
    agree = float((np.sign(diff_signed[sel]) == np.sign(g_signed[sel])).mean()) if sel.any() else float("nan")
    return {"ratio": ratio, "corr": corr, "agree": agree,
            "d_in": float(inside.mean()), "d_out": float(outside.mean()), "mask_frac": float(mask.mean())}


def main():
    args = parse_args()
    run_dir = Path(args.run).resolve()
    probe_out = Path(args.probe_out).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve(); cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    layers = [s.strip() for s in args.layers.split(",")] if args.layers else (["11", "final"] if args.family == "yak" else ["13", "final"])
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "weights" / "step_021700.pt"
    torch.manual_seed(args.seed)
    print(f"[diff] run={run_dir} ckpt={ckpt} family={args.family} layers={layers} n={args.heldout_samples}", flush=True)
    t0 = time.time()
    cfg, model, processor, dtype = FP.AV.build(run_dir, ckpt, device)
    model.eval()
    print(f"[diff] model loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    meta = FP.run_extract(args, model, cfg, run_dir, out, cache, layers, device)

    he = meta["heldout"]; n = int(he["count"]); grid = tuple(he["grid"]); patch = int(he["patch"])
    fut = np.load(cache / "heldout_future.npy", mmap_mode="r")[:n]
    cur = np.load(cache / "heldout_current.npy", mmap_mode="r")[:n]
    rows_meta = he["rows"][:n]

    sd = torch.load(probe_out / "readouts.pt", map_location="cpu")
    results = {}
    preds = {}
    for l in layers:
        feats = np.load(cache / f"heldout_feats_{l}.npy", mmap_mode="r")[:n]
        d_in = int(feats.shape[-1])
        for t in ("future", "current"):
            r = FP.PatchReadout(d_in, args.probe_hidden, patch).to(device)
            r.load_state_dict(sd[f"{l}|{t}"]); r.eval()
            preds[(l, t)] = FP.predict_images(r, feats, grid, patch, device)
        print(f"[diff] layer {l}: decoded both readouts", flush=True)

    rng = np.random.default_rng(args.seed)
    shuffle = rng.permutation(n)
    for l in layers:
        rows = []
        t0 = time.time()
        for i in range(n):
            c = cur[i].astype(np.float32) / 127.5 - 1.0
            f = fut[i].astype(np.float32) / 127.5 - 1.0
            pf = preds[(l, "future")][i].astype(np.float32) / 127.5 - 1.0
            pc = preds[(l, "current")][i].astype(np.float32) / 127.5 - 1.0
            mask = FP.change_mask(cur[i], fut[i])
            if mask.sum() < 10 or (~mask).sum() < 10:
                continue
            r = stats_one(pf - pc, f - c, mask)
            j = int(shuffle[i])
            mj = FP.change_mask(cur[j], fut[j])
            if mj.sum() >= 10 and (~mj).sum() >= 10:
                fj = fut[j].astype(np.float32) / 127.5 - 1.0
                cj = cur[j].astype(np.float32) / 127.5 - 1.0
                nl = stats_one(pf - pc, fj - cj, mj)
                r.update({f"null_{k}": v for k, v in nl.items() if k in ("ratio", "corr", "agree")})
            r.update({"pos": i, "suite": rows_meta[i]["suite"]})
            rows.append(r)
            if i % 500 == 0:
                print(f"[diff] L{l} {i}/{n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        keys = [k for k in ("ratio", "corr", "agree", "d_in", "d_out", "mask_frac", "null_ratio", "null_corr", "null_agree") if k in rows[0]]
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        agg["n"] = len(rows)
        suites = sorted({r["suite"] for r in rows})
        agg["by_suite"] = {s: {k: float(np.nanmean([r[k] for r in rows if r["suite"] == s])) for k in ("ratio", "corr", "agree")} for s in suites}
        results[l] = {"aggregate": agg, "rows": rows}
        print(f"[diff] L{l}: ratio {agg['ratio']:.2f}x (null {agg.get('null_ratio', float('nan')):.2f}x)  "
              f"corr {agg['corr']:+.3f} (null {agg.get('null_corr', float('nan')):+.3f})  "
              f"agree {100 * agg['agree']:.1f}% (null {100 * agg.get('null_agree', float('nan')):.1f}%)  n={agg['n']}", flush=True)

    (out / "diff_summary.json").write_text(json.dumps({l: results[l]["aggregate"] for l in layers}, indent=1))
    (out / "diff_rows.json").write_text(json.dumps({l: results[l]["rows"] for l in layers}))

    # visualisation: current, future, |true change|, |readout difference|
    def heat(x):
        x = np.clip(x / max(x.max(), 1e-6), 0, 1)
        return (np.stack([np.clip(2.2 * x, 0, 1), np.clip(2.2 * x - 0.8, 0, 1), np.clip(3.0 * x - 2.0, 0, 1)], -1) * 255).astype(np.uint8)
    vdir = out / "vis"; vdir.mkdir(exist_ok=True)
    if args.vis_positions:
        picked = [int(v) for v in args.vis_positions.split(",") if v.strip() != ""]
        picked = [i for i in picked if 0 <= i < n]
        print(f"[diff] rendering requested positions {picked}", flush=True)
    else:
        per = {}; picked = []
        for i in range(n):
            s = rows_meta[i]["suite"]
            if per.get(s, 0) < args.vis_per_suite:
                per[s] = per.get(s, 0) + 1; picked.append(i)
    l0 = layers[0]
    for i in picked:
        tag = f"{i:04d}_{rows_meta[i]['suite']}"
        g = np.abs(fut[i].astype(np.float32) - cur[i].astype(np.float32)).mean(-1) / 255.0
        dm = np.abs(preds[(l0, 'future')][i].astype(np.float32) - preds[(l0, 'current')][i].astype(np.float32)).mean(-1) / 255.0
        Image.fromarray(cur[i]).save(vdir / f"{tag}_current.png")
        Image.fromarray(fut[i]).save(vdir / f"{tag}_future.png")
        Image.fromarray(heat(g)).save(vdir / f"{tag}_true_change.png")
        Image.fromarray(heat(dm)).save(vdir / f"{tag}_readout_diff_L{l0}.png")
        # signed maps for a diverging visualisation: what gets brighter (+) and darker (-).
        # Saved raw so that several policies can later share one symmetric scale.
        g_s = (fut[i].astype(np.float32) - cur[i].astype(np.float32)).mean(-1) / 255.0
        d_s = (preds[(l0, 'future')][i].astype(np.float32) - preds[(l0, 'current')][i].astype(np.float32)).mean(-1) / 255.0
        np.savez_compressed(vdir / f"{tag}_signed_L{l0}.npz", true_change=g_s.astype(np.float32),
                            readout_diff=d_s.astype(np.float32), current=cur[i], future=fut[i])
    (vdir / "index.json").write_text(json.dumps([{"pos": i, "suite": rows_meta[i]["suite"], "instruction": rows_meta[i]["instruction"]} for i in picked], indent=1))
    print("DIFF_STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
