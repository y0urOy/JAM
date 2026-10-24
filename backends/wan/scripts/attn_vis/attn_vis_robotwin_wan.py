"""Action-to-visual attention of Fast-WAM / JAM (Wan) RoboTwin policies on stored demonstration frames.

The policy is built exactly as the RoboTwin evaluator builds it (experiments/robotwin/jam_policy
deploy_policy.get_model with the strict-evaluation settings) and each record is fed through
`_infer_action_chunk`.  The MoT mixed-attention kernel is wrapped during the cached action pass: keys are
[first-frame video tokens (12x10: the 384x320 three-camera layout after the 16x VAE and the 2x2 patch, head
camera = rows 0-7) | action tokens], so the action-to-visual attention is the softmax mass on the first 120 keys,
averaged over heads and action queries, per layer and denoising step.  Output npz: maps [L, S, 1, 12, 10].
"""
import argparse, json, os, sys, types
from pathlib import Path
import numpy as np, torch
from einops import rearrange
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO, str(REPO / "experiments" / "robotwin")):
    sys.path.insert(0, str(p))
import jam_policy.deploy_policy as dp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Hydra task config, e.g. robotwin_uncond_3cam_384_1e-4")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    usr = dict(sim_cfg_name="sim_robotwin.yaml", sim_task=args.task, ckpt_setting=str(args.ckpt), dataset_stats_path=str(args.stats),
               device="cuda", mixed_precision="bf16", action_horizon=None, replan_steps=24, num_inference_steps=args.steps,
               sigma_shift=5.0, seed=args.seed, text_cfg_scale=1.0, rand_device="cpu", cache_text_context=False)
    policy = dp.get_model(usr); model = policy.model; mot = model.mot
    H = int(mot.num_heads); horizon = int(policy.action_horizon)
    GH, GW = 384 // 32, 320 // 32; NV = GH * GW      # 16x VAE, then 2x2 patches
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = {"maps": {}, "groups": {}, "count": {}}
    orig = mot._mixed_attention

    seen = {}

    def mixed(q_cat, k_cat, v_cat, attention_mask):
        y = orig(q_cat, k_cat, v_cat, attention_mask)
        shp = (int(q_cat.shape[1]), int(k_cat.shape[1]))
        if shp not in seen:
            seen[shp] = True; print("mixed-attention call shapes (q_len, k_len):", shp, flush=True)
        if int(q_cat.shape[1]) == horizon and int(k_cat.shape[1]) == NV + horizon:
            q = rearrange(q_cat, "b s (n d) -> b n s d", n=H).float(); k = rearrange(k_cat, "b s (n d) -> b n s d", n=H).float()
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            m = attention_mask.to(device=q.device, dtype=torch.bool)
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            row = probs.mean(dim=(0, 1, 2))                       # [NV + horizon]
            layer = rec["count"].get("layer", 0); rec["count"]["layer"] = (layer + 1) % mot.num_layers
            step = rec["count"].get(layer, 0); rec["count"][layer] = step + 1
            rec["maps"][(layer, step)] = row[:NV].reshape(1, GH, GW).cpu().numpy()
            rec["groups"][(layer, step)] = np.array([0.0, float(row[:NV].sum()), float(row[NV:].sum()), 0.0], dtype=np.float32)
        return y

    mot._mixed_attention = mixed
    for r in json.load(open(args.manifest)):
        rec["maps"].clear(); rec["groups"].clear(); rec["count"].clear()
        obs = {"observation": {k + "_camera": {"rgb": np.asarray(Image.open(r[k]).convert("RGB"))} for k in ("head", "left", "right")},
               "joint_action": {"vector": np.asarray(r["state"], dtype=np.float32)}}
        with torch.inference_mode():
            action = policy._infer_action_chunk(obs, r["instruction"])
        layers = sorted({k[0] for k in rec["maps"]}); steps = sorted({k[1] for k in rec["maps"]})
        if not rec["maps"]:
            raise RuntimeError(f"no attention captured: horizon={horizon}, expected k_len={NV + horizon}, seen={list(seen)}")
        stack = np.stack([[rec["maps"][(l, s)] for s in steps] for l in layers])
        gmass = np.stack([[rec["groups"][(l, s)] for s in steps] for l in layers])
        np.savez_compressed(out / f"{r['tag']}.npz", maps=stack, layers=np.array(layers), steps=np.array(steps),
                            head=np.asarray(Image.open(r["head"]).convert("RGB")), layout="fastwam_384x320_12x10", family="wan",
                            instruction=r["instruction"], action=np.asarray(action), group_mass=gmass,
                            group_names=np.array(["text", "visual", "action", "other"]), record=json.dumps({k: v for k, v in r.items() if k != "state"}))
        print(f"[wan] {r['tag']}: layers={len(layers)} steps={len(steps)} visual share={gmass[..., 1].mean():.3f} max={stack.mean(axis=(0, 1)).max():.4f}", flush=True)


if __name__ == "__main__":
    main()
