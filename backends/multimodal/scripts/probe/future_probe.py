#!/usr/bin/env python3
"""Future-decodability probe for ImageWAM and JAM policies (Video-Mirai Figure 1 protocol).

For one trained policy: freeze it, run the deployed current-observation pathway on LIBERO
windows, cache per-token hidden states at chosen layers, train a small per-token MLP readout
that reconstructs the RGB frame 16 steps ahead (and, as a reference, the current frame),
and evaluate on held-out episodes against a copy-current baseline.  For the native
generative ImageWAM policy, its own generated future image is evaluated the same way.

Families
  yak  : Ovis-U1 policy (native ImageWAM generation or JAM).  Tokens = 2x2-packed Ovis-VAE
         patches of the current 224x448 frame (14x28 grid, 16 px per token, 1536-d);
         layers are 1-indexed Yak blocks (6 double + 12 single), captured on the
         action-inference pathway (no future tokens in the stream).
  qwen : Qwen3.5-2B JAM.  Tokens = image placeholders of the two 224x224 views
         (7x7 per view, 32 px per token, 2048-d); layers are decoder layers,
         and "final" is the normalised last hidden state.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "attn_vis"))
import attn_vis as AV  # noqa: E402  (model loading helpers shared with the attention figure)
from jam_multimodal.utils import misc  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


# ----------------------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.yaml + dataset_stats.json")
    ap.add_argument("--ckpt", default=None, help="weights checkpoint (default: checkpoints/weights/step_021700.pt)")
    ap.add_argument("--family", required=True, choices=["yak", "qwen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True, help="scratch directory for feature memmaps")
    ap.add_argument("--layers", default=None, help="comma list; 'final' allowed (default yak: 11,final  qwen: 13,final)")
    ap.add_argument("--train-samples", type=int, default=20000)
    ap.add_argument("--heldout-samples", type=int, default=2000)
    ap.add_argument("--val-proportion", type=float, default=0.05, help="episode-level held-out fraction")
    ap.add_argument("--select-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--probe-batch", type=int, default=32)
    ap.add_argument("--probe-hidden", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--generate", type=int, default=0, help="native only: generate the future for the first N held-out samples")
    ap.add_argument("--gen-steps", type=int, default=20)
    ap.add_argument("--vis-per-suite", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all", choices=["all", "extract", "probe"])
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ----------------------------------------------------------------------------- dataset
def build_dataset(cfg, run_dir: Path, heldout: bool, val_proportion: float):
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    data_cfg.video_augmentation = None
    data_cfg.val_set_proportion = float(val_proportion)
    data_cfg.is_training_set = not heldout
    data_cfg.skip_padding_as_possible = False
    stats = run_dir / "dataset_stats.json"
    if stats.is_file():
        data_cfg.pretrained_norm_stats = str(stats)
    elif data_cfg.get("pretrained_norm_stats"):
        p = Path(str(data_cfg.pretrained_norm_stats))
        data_cfg.pretrained_norm_stats = str(p if p.is_absolute() else (ROOT / p).resolve())
    dirs = []
    for d in data_cfg.dataset_dirs:
        p = Path(str(d))
        dirs.append(str(p if p.is_absolute() else (ROOT / p).resolve()))
    data_cfg.dataset_dirs = dirs
    for key in ("text_embedding_cache_dir", "qwen_text_cache_dir"):
        if key in data_cfg and data_cfg[key]:
            p = Path(str(data_cfg[key]))
            data_cfg[key] = str(p if p.is_absolute() else (ROOT / p).resolve())
    return instantiate(data_cfg)


def suite_boundaries(ds):
    try:
        subs = ds.lerobot_dataset.multi_dataset._datasets
        bounds, start = [], 0
        for name, sub in zip(SUITES, subs):
            end = start + int(sub.num_frames)
            bounds.append((name, start, end))
            start = end
        if start != len(ds):
            print(f"WARNING suite boundaries {start} != len(ds) {len(ds)}; suites unknown", flush=True)
            return None
        return bounds
    except Exception as e:  # noqa: BLE001
        print(f"WARNING cannot derive suite boundaries: {e}", flush=True)
        return None


def suite_of(idx, bounds):
    if bounds is None:
        return "unknown"
    for name, s, e in bounds:
        if s <= idx < e:
            return name
    return "unknown"


class Indexed(torch.utils.data.Dataset):
    def __init__(self, ds, indices):
        self.ds, self.indices = ds, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        s = self.ds[idx]
        s["_idx"] = torch.tensor(idx)
        return s


def collate(items):
    out = {}
    for k in items[0]:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return out


def video_to_float(video):
    if video.dtype == torch.uint8:
        return video.float() / 127.5 - 1.0
    return video.float()


def to_u8(img):  # [B,3,H,W] in [-1,1] -> [B,H,W,3] uint8
    return ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


# ----------------------------------------------------------------------------- feature extraction
class Extractor:
    def __init__(self, model, family, layers, device):
        self.model, self.family, self.device = model, family, device
        self.layers = layers  # list of str ('11', 'final', ...)

    @torch.inference_mode()
    def __call__(self, cur, instructions, proprio):
        """cur [B,3,H,W] in [-1,1] (cpu float); returns dict layer -> [B, gh*gw, D] fp16 cpu, and grid."""
        if self.family == "yak":
            return self._yak(cur, instructions, proprio)
        return self._qwen(cur, instructions)

    def _yak(self, cur, instructions, proprio):
        m = self.model
        cur = cur.to(device=m.device, dtype=m.torch_dtype)
        text_hidden, text_mask = m._encode_ovis_u1_condition_online(list(instructions), cur)
        text_hidden, text_mask = m._append_proprio_to_context_if_enabled(
            context=text_hidden, context_mask=text_mask, proprio=proprio.to(device=m.device), source="probe")
        ref_tokens, ref_ids = m._encode_ovis_u1_image_tokens(cur, time_value=1.0)
        b = int(ref_tokens.shape[0])
        empty = torch.empty(b, 0, int(m.video_expert.transformer.in_channels), device=m.device, dtype=m.torch_dtype)
        empty_ids = torch.empty(b, 0, 3, device=m.device, dtype=m.torch_dtype)
        pre = m.video_expert.pre_dit(x=empty, timestep=torch.zeros(b, device=m.device, dtype=m.torch_dtype),
                                     context=text_hidden, context_mask=text_mask, ref_image_hidden_states=ref_tokens,
                                     target_img_ids=empty_ids, ref_img_ids=ref_ids)
        mask = m._build_mot_attention_mask_ovis_u1(batch_size=b, txt_len=int(pre["txt_len"]), target_len=0,
                                                   cond_len=int(pre["cond_len"]), action_len=0, device=m.device,
                                                   text_attention_mask=pre["text_mask"])
        total = int(m.video_expert.double_layers) + int(m.video_expert.single_layers)
        want = {total if l == "final" else int(l) for l in self.layers}
        out = m.mot.forward_yak_video_only(video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
                                           attention_mask=mask, capture_layers=want)
        H, W = int(cur.shape[-2]), int(cur.shape[-1])
        grid = (H // 16, W // 16)
        feats = {}
        for l in self.layers:
            li = total if l == "final" else int(l)
            h = out["captured"][li]
            if h.shape[1] != grid[0] * grid[1]:
                raise RuntimeError(f"yak token count {h.shape[1]} != grid {grid}")
            feats[l] = h.to(torch.float16).cpu()
        return feats, grid, 16

    def _qwen(self, cur, instructions):
        import jam_multimodal.models.backbones.jam_qwen35 as qm
        m = self.model
        cur = cur.to(device=m.device, dtype=m.torch_dtype)
        views = m._split_views(cur)
        texts = m._student_texts(list(instructions))
        model_inputs = m._prepare_qwen_inputs(views, texts)
        token_indices = model_inputs.pop("_image_token_indices", None)
        if m.action_fusion_type == qm.QWEN_ACTION_FUSION_NATIVE_QWEN:
            position_ids = model_inputs.get("position_ids")
            if position_ids is None:
                position_ids = m.student_backbone.compute_3d_position_ids(
                    input_ids=model_inputs["input_ids"], inputs_embeds=None,
                    image_grid_thw=model_inputs["image_grid_thw"], video_grid_thw=None,
                    attention_mask=model_inputs["attention_mask"], past_key_values=None,
                    mm_token_type_ids=model_inputs["mm_token_type_ids"])
            model_inputs["position_ids"] = position_ids
        output, _last, dec_out = qm.forward_with_decoder_layer_capture(
            m.student_backbone, layer=None, model_inputs=model_inputs, return_decoder_outputs=True)
        side = int(math.isqrt(int(m.visual_tokens_per_view)))
        nv = int(m.num_views)
        grid = (side, side * nv)
        feats = {}
        for l in self.layers:
            h = output.last_hidden_state if l == "final" else dec_out[int(l) - 1]
            g = qm.extract_grouped_visual_tokens(h, model_inputs["input_ids"], image_token_id=m.image_token_id,
                                                 num_images_per_sample=nv, visual_tokens_per_image=int(m.visual_tokens_per_view),
                                                 token_indices=token_indices)  # [B, V, side*side, D]
            b, _, _, d = g.shape
            g = g.reshape(b, nv, side, side, d).permute(0, 2, 1, 3, 4).reshape(b, side, nv * side, d)  # global row-major grid
            feats[l] = g.reshape(b, side * nv * side, d).to(torch.float16).cpu()
        patch = int(cur.shape[-2]) // side
        return feats, grid, patch


def run_extract(args, model, cfg, run_dir, out, cache, layers, device):
    misc.register_work_dir(str(out))
    plan = [("train", args.train_samples, False), ("heldout", args.heldout_samples, True)]
    meta_all = {}
    for split, n_target, heldout in plan:
        ds = build_dataset(cfg, run_dir, heldout=heldout, val_proportion=args.val_proportion)
        bounds = suite_boundaries(ds)
        rng = np.random.default_rng(args.select_seed + (1 if heldout else 0))
        order = rng.permutation(len(ds))
        need = min(n_target, len(ds))
        candidates = order[: min(len(ds), int(need * 1.25) + 64)]
        print(f"[{split}] dataset len={len(ds)} target={need} candidates={len(candidates)} bounds={bounds}", flush=True)
        loader = torch.utils.data.DataLoader(Indexed(ds, candidates), batch_size=args.batch_size, num_workers=args.workers,
                                             shuffle=False, collate_fn=collate, pin_memory=False, persistent_workers=False)
        ext = Extractor(model, args.family, layers, device)
        feats_mm = {}
        fut_mm = cur_mm = None
        rows = []
        n = 0
        t0 = time.time()
        for bi, batch in enumerate(loader):
            video = video_to_float(batch["video"])  # [B,3,T,H,W]
            if video.ndim != 5 or video.shape[2] < 2:
                raise RuntimeError(f"unexpected video shape {tuple(video.shape)}")
            pad = batch["image_is_pad"]
            keep = ~(pad.reshape(pad.shape[0], -1).any(dim=1))
            if not bool(keep.any()):
                continue
            ki = keep.nonzero(as_tuple=True)[0]
            cur, fut = video[ki, :, 0], video[ki, :, -1]
            instr = [batch["instruction"][int(i)] for i in ki]
            proprio = batch["proprio"][ki]
            feats, grid, patch = ext(cur, instr, proprio)
            b = int(cur.shape[0])
            if fut_mm is None:
                H, W = int(cur.shape[-2]), int(cur.shape[-1])
                for l, f in feats.items():
                    feats_mm[l] = np.lib.format.open_memmap(str(cache / f"{split}_feats_{l}.npy"), mode="w+", dtype=np.float16,
                                                            shape=(need, f.shape[1], f.shape[2]))
                fut_mm = np.lib.format.open_memmap(str(cache / f"{split}_future.npy"), mode="w+", dtype=np.uint8, shape=(need, H, W, 3))
                cur_mm = np.lib.format.open_memmap(str(cache / f"{split}_current.npy"), mode="w+", dtype=np.uint8, shape=(need, H, W, 3))
            take = min(b, need - n)
            for l, f in feats.items():
                feats_mm[l][n:n + take] = f[:take].numpy()
            fut_mm[n:n + take] = to_u8(fut[:take])
            cur_mm[n:n + take] = to_u8(cur[:take])
            for j in range(take):
                idx = int(batch["_idx"][int(ki[j])])
                row = {"pos": n + j, "index": idx, "suite": suite_of(idx, bounds), "instruction": instr[j],
                       "proprio0": [float(v) for v in proprio[j].reshape(proprio.shape[1], -1)[0].tolist()]}
                for key in ("episode_index", "frame_index", "dataset_index"):
                    if key in batch:
                        row[key] = int(batch[key][int(ki[j])])
                rows.append(row)
            n += take
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"[{split}] {n}/{need} samples  {n / max(el, 1e-6):.1f} samples/s  elapsed {el / 60:.1f} min", flush=True)
            if n >= need:
                break
        if n < need:
            print(f"WARNING [{split}] only {n} of {need} samples collected", flush=True)
        for mm in list(feats_mm.values()) + [fut_mm, cur_mm]:
            mm.flush()
        meta_all[split] = {"count": n, "grid": list(grid), "patch": patch, "rows": rows,
                           "feat_shapes": {l: list(feats_mm[l].shape) for l in feats_mm}}
        print(f"[{split}] done: {n} samples in {(time.time() - t0) / 60:.1f} min", flush=True)
        del loader, ds
    meta_all["layers"] = layers
    meta_all["family"] = args.family
    meta_all["run"] = str(run_dir)
    (out / "extract_meta.json").write_text(json.dumps(meta_all, indent=1))
    print("PROBE_STAGE_DONE extract", flush=True)
    return meta_all


# ----------------------------------------------------------------------------- readout
class PatchReadout(nn.Module):
    def __init__(self, d_in, hidden, patch):
        super().__init__()
        self.patch = patch
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 3 * patch * patch))

    def forward(self, x):
        return self.net(x)


def img_to_patches(img_u8, grid, patch, device):
    """uint8 [B,H,W,3] numpy -> float [-1,1] patches [B, gh*gw, 3*p*p] (row-major grid)."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    b = x.shape[0]
    gh, gw = grid
    x = x.reshape(b, 3, gh, patch, gw, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, 3 * patch * patch)
    return x


def patches_to_img(p, grid, patch):
    """[B, gh*gw, 3*p*p] -> [B,3,H,W] float."""
    b = p.shape[0]
    gh, gw = grid
    x = p.reshape(b, gh, gw, 3, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, gh * patch, gw * patch)
    return x


def train_readouts(args, meta, cache, out, device):
    layers = meta["layers"]
    tr = meta["train"]
    n_train, grid, patch = int(tr["count"]), tuple(tr["grid"]), int(tr["patch"])
    feats = {l: np.load(cache / f"train_feats_{l}.npy", mmap_mode="r") for l in layers}
    fut = np.load(cache / "train_future.npy", mmap_mode="r")
    cur = np.load(cache / "train_current.npy", mmap_mode="r")
    d_in = int(feats[layers[0]].shape[-1])
    torch.manual_seed(args.seed)
    readouts = {(l, t): PatchReadout(d_in, args.probe_hidden, patch).to(device) for l in layers for t in ("future", "current")}
    params = [p for r in readouts.values() for p in r.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, n_train // args.probe_batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    rng = np.random.default_rng(args.seed)
    print(f"[probe] train n={n_train} grid={grid} patch={patch} d_in={d_in} readouts={list(readouts)} steps={total_steps}", flush=True)
    t0 = time.time()
    step = 0
    hist = []
    for ep in range(args.epochs):
        perm = rng.permutation(n_train)
        run_loss = {k: 0.0 for k in readouts}
        cnt = 0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * args.probe_batch:(s + 1) * args.probe_batch])
            tgt = {"future": img_to_patches(fut[idx], grid, patch, device), "current": img_to_patches(cur[idx], grid, patch, device)}
            loss_total = 0.0
            for l in layers:
                x = torch.from_numpy(np.ascontiguousarray(feats[l][idx])).to(device).float()
                for t in ("future", "current"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = readouts[(l, t)](x)
                    loss = F.mse_loss(pred.float(), tgt[t])
                    loss_total = loss_total + loss
                    run_loss[(l, t)] += float(loss.item())
            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            cnt += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"[probe] ep {ep} step {step}/{total_steps} " + " ".join(f"{l}/{t}={v / cnt:.4f}" for (l, t), v in run_loss.items()) + f" ({el / 60:.1f} min)", flush=True)
        hist.append({"epoch": ep, **{f"{l}/{t}": v / max(cnt, 1) for (l, t), v in run_loss.items()}})
        print(f"[probe] epoch {ep} done: " + " ".join(f"{l}/{t}={v / max(cnt, 1):.4f}" for (l, t), v in run_loss.items()), flush=True)
    torch.save({f"{l}|{t}": r.state_dict() for (l, t), r in readouts.items()}, out / "readouts.pt")
    (out / "train_history.json").write_text(json.dumps(hist, indent=1))
    print(f"[probe] training done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return readouts


# ----------------------------------------------------------------------------- metrics
def change_mask(cur_u8, fut_u8, thr=0.10, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    diff = np.abs(cur_u8.astype(np.float32) - fut_u8.astype(np.float32)).mean(-1) / 255.0
    return gaussian_filter(diff, sigma) > thr


def psnr(a_u8, b_u8):
    mse = np.mean((a_u8.astype(np.float32) - b_u8.astype(np.float32)) ** 2)
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a_u8, b_u8):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a_u8, b_u8, channel_axis=-1, data_range=255))


def region_metrics(pred_u8, cur_u8, fut_u8, mask):
    p, c, f = (x.astype(np.float32) / 255.0 for x in (pred_u8, cur_u8, fut_u8))
    e_pf, e_pc, e_cf = (np.abs(p - f).mean(-1), np.abs(p - c).mean(-1), np.abs(c - f).mean(-1))
    static = ~mask
    r = {"mask_frac": float(mask.mean()),
         "psnr": psnr(pred_u8, fut_u8), "ssim": ssim(pred_u8, fut_u8),
         "static_l1_vs_future": float(e_pf[static].mean()) if static.any() else float("nan")}
    if mask.any():
        r.update({"changed_l1_vs_future": float(e_pf[mask].mean()), "changed_l1_vs_current": float(e_pc[mask].mean()),
                  "changed_frac_closer_future": float((e_pf[mask] < e_pc[mask]).mean())})
    else:
        r.update({"changed_l1_vs_future": float("nan"), "changed_l1_vs_current": float("nan"), "changed_frac_closer_future": float("nan")})
    r["copy_changed_l1_vs_future"] = float(e_cf[mask].mean()) if mask.any() else float("nan")
    return r


def aggregate(rows, key_pred, suites):
    """rows: list of dicts with 'suite' and metric dict under key_pred."""
    out = {}
    fields = ["psnr", "ssim", "changed_l1_vs_future", "changed_l1_vs_current", "changed_frac_closer_future", "static_l1_vs_future", "mask_frac"]
    groups = {"all": rows}
    for s in suites:
        groups[s] = [r for r in rows if r["suite"] == s]
    for g, rs in groups.items():
        vals = {f: float(np.nanmean([r[key_pred][f] for r in rs])) if rs else float("nan") for f in fields}
        vals["n"] = len(rs)
        out[g] = vals
    return out


# ----------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def predict_images(readout, feats_np, grid, patch, device, bs=64):
    outs = []
    for s in range(0, feats_np.shape[0], bs):
        x = torch.from_numpy(np.ascontiguousarray(feats_np[s:s + bs])).to(device).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            p = readout(x)
        img = patches_to_img(p.float(), grid, patch)
        outs.append(to_u8(img))
    return np.concatenate(outs, 0)


def run_eval(args, model, meta, readouts, cache, out, device):
    layers = meta["layers"]
    he = meta["heldout"]
    n, grid, patch = int(he["count"]), tuple(he["grid"]), int(he["patch"])
    fut = np.load(cache / "heldout_future.npy", mmap_mode="r")[:n]
    cur = np.load(cache / "heldout_current.npy", mmap_mode="r")[:n]
    rows_meta = he["rows"][:n]
    preds = {}
    for l in layers:
        feats = np.load(cache / f"heldout_feats_{l}.npy", mmap_mode="r")[:n]
        for t in ("future", "current"):
            preds[f"probe_{t}_L{l}"] = predict_images(readouts[(l, t)], feats, grid, patch, device)
    # native generation
    gen = None
    if args.generate > 0:
        gen_n = min(args.generate, n)
        gen = np.zeros((gen_n, fut.shape[1], fut.shape[2], 3), np.uint8)
        t0 = time.time()
        for i in range(gen_n):
            x = torch.from_numpy(cur[i].astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(device=model.device, dtype=model.torch_dtype)
            prop = torch.from_numpy(np.asarray(rows_meta[i]["proprio0"], np.float32))[None].to(device=model.device, dtype=model.torch_dtype) if "proprio0" in rows_meta[i] else None
            res = model.infer_video_ovis_u1(prompt=rows_meta[i]["instruction"], input_image=x, proprio=prop,
                                            num_inference_steps=args.gen_steps, seed=args.seed)
            gen[i] = to_u8(res["image"][None].float())[0]
            if i % 25 == 0:
                print(f"[gen] {i}/{gen_n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.save(out / "generated_heldout.npy", gen)
        print(f"[gen] done {gen_n} images in {(time.time() - t0) / 60:.1f} min", flush=True)
    # metrics
    rows = []
    t0 = time.time()
    for i in range(n):
        mask = change_mask(cur[i], fut[i])
        r = {"pos": i, "suite": rows_meta[i]["suite"], "index": rows_meta[i]["index"]}
        r["copy_current"] = region_metrics(cur[i], cur[i], fut[i], mask)
        for k, arr in preds.items():
            ref = fut[i] if "future" in k else cur[i]
            r[k] = region_metrics(arr[i], cur[i], ref, mask) if "future" in k else {"psnr": psnr(arr[i], cur[i]), "ssim": ssim(arr[i], cur[i]),
                                                                                    "changed_l1_vs_future": float("nan"), "changed_l1_vs_current": float("nan"),
                                                                                    "changed_frac_closer_future": float("nan"), "static_l1_vs_future": float("nan"), "mask_frac": float(mask.mean())}
        if gen is not None and i < gen.shape[0]:
            r["generated"] = region_metrics(gen[i], cur[i], fut[i], mask)
        rows.append(r)
        if i % 500 == 0:
            print(f"[eval] {i}/{n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    suites = sorted({r["suite"] for r in rows})
    summary = {"family": meta["family"], "run": meta["run"], "layers": layers, "n_heldout": n, "grid": list(grid), "patch": patch,
               "predictors": {}}
    keys = ["copy_current"] + list(preds) + (["generated"] if gen is not None else [])
    for k in keys:
        rs = [r for r in rows if k in r]
        summary["predictors"][k] = aggregate(rs, k, suites)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "metrics_heldout.json").write_text(json.dumps(rows))
    # visualisation subset: first vis_per_suite held-out samples per suite (deterministic selection order)
    vis = []
    per = {}
    for i in range(n):
        s = rows_meta[i]["suite"]
        if per.get(s, 0) < args.vis_per_suite:
            per[s] = per.get(s, 0) + 1
            vis.append(i)
    vdir = out / "vis"
    vdir.mkdir(exist_ok=True)
    from PIL import Image
    index = []
    for i in vis:
        tag = f"{i:04d}_{rows_meta[i]['suite']}"
        Image.fromarray(cur[i]).save(vdir / f"{tag}_current.png")
        Image.fromarray(fut[i]).save(vdir / f"{tag}_future.png")
        for k, arr in preds.items():
            Image.fromarray(arr[i]).save(vdir / f"{tag}_{k}.png")
        if gen is not None and i < gen.shape[0]:
            Image.fromarray(gen[i]).save(vdir / f"{tag}_generated.png")
        index.append({"pos": i, "tag": tag, **{k: v for k, v in rows_meta[i].items() if k != "instruction"}, "instruction": rows_meta[i]["instruction"]})
    (vdir / "index.json").write_text(json.dumps(index, indent=1))
    # console summary
    print("\n=== held-out summary (all suites) ===")
    hdr = f"{'predictor':28s}{'PSNR':>7s}{'SSIM':>7s}{'chg L1 fut':>11s}{'chg L1 cur':>11s}{'closer fut':>11s}{'static L1':>10s}{'n':>6s}"
    print(hdr)
    for k in keys:
        a = summary["predictors"][k]["all"]
        print(f"{k:28s}{a['psnr']:7.2f}{a['ssim']:7.3f}{a['changed_l1_vs_future']:11.3f}{a['changed_l1_vs_current']:11.3f}{100 * a['changed_frac_closer_future']:10.1f}%{a['static_l1_vs_future']:10.3f}{a['n']:6d}")
    print("PROBE_STAGE_DONE eval", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    run_dir = Path(args.run).resolve()
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "weights" / "step_021700.pt"
    out, cache = Path(args.out), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    layers = (args.layers.split(",") if args.layers else (["11", "final"] if args.family == "yak" else ["13", "final"]))
    layers = [l.strip() for l in layers]
    device = args.device
    torch.manual_seed(args.seed)
    print(f"PROBE_START family={args.family} run={run_dir} ckpt={ckpt} layers={layers} out={out} cache={cache}", flush=True)
    t0 = time.time()
    cfg, model, processor, dtype = AV.build(run_dir, ckpt, device)
    model.eval()
    print(f"model loaded in {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.stage in ("all", "extract"):
        meta = run_extract(args, model, cfg, run_dir, out, cache, layers, device)
        # record the normalised proprio of held-out samples for generation (first step)
    else:
        meta = json.loads((out / "extract_meta.json").read_text())
    if args.stage == "extract":
        return
    # proprio for generation: re-read from the cached rows if stored
    readouts = train_readouts(args, meta, cache, out, device)
    run_eval(args, model, meta, readouts, cache, out, device)
    print(f"PROBE_COMPLETE total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
