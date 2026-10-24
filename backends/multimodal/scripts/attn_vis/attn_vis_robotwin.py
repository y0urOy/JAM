"""Action-to-visual attention maps for RoboTwin policies on stored demonstration frames.

Families:
  flux2   FLUX.2 backbone + MoT action expert (released ImageWAM FLUX.2 weights, or the JAM (FLUX.2)
          checkpoint).  The FLUX.2 inference path implements the `action_attention_capture` protocol itself:
          keys per call are [text (+proprio) | image tokens (18x16 over the compact 288x256 layout) | action].
  qwen    Qwen3.5-2B + native causal ActionQwen24 (three 224x224 views, 7x7 image tokens each).

The policy object is built exactly as the RoboTwin evaluator builds it (deploy_policy.get_model with the Hydra
task config and the FLUX2_OVERRIDES / QWEN_OVERRIDES model overrides below), and each record is fed
through `_infer_action_chunk`, i.e. the deployed preprocessing and inference loop.  Maps are averaged over heads
and action queries and stored per layer and denoising step: maps[L, S, V, gh, gw], group_mass[L, S, 4].
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO, REPO / "scripts" / "attn_vis"):
    sys.path.insert(0, str(p))
from attn_vis import Recorder, MoTCapture, patch_qwen  # noqa: E402
import experiments.robotwin.jam_policy.deploy_policy as dp  # noqa: E402

FLUX2_OVERRIDES = [
    "model.flux2_src_path={flux2_src}", "model.flux2_model_path={flux2_model}", "model.ae_model_path={ae_model}",
    "model.variant=klein-base-4b", "model.qwen3_model_spec={qwen3}", "model.qwen_context_len=512",
    "model.pack_proprio_after_text=true", "model.mot_checkpoint_mixed_attn=false", "model.load_text_encoder=true",
    "model.proprio_dim=14",
]
QWEN_OVERRIDES = [
    "model.model_id={qwen}", "model.teacher_model_id={qwen}", "model.num_views=3",
    "model.skip_dit_load_from_pretrain=false", "model.action_dit_pretrained_path=null",
    "model.foresight.load_teacher=false", "model.foresight.teacher_early_exit=true", "model.foresight.layer=13",
    "model.foresight.frame_offsets=[16]", "model.load_text_encoder=true", "model.proprio_dim=14",
]


class GridCapture(MoTCapture):
    """MoTCapture over one rectangular token grid (no view split)."""

    def update(self, attn_probs, *, layer_idx, block_type, prefix_len, action_len):
        a, b = int(self.cond[0]), int(self.cond[1])
        vis = attn_probs[:, :, :, a:b].float().mean(dim=(0, 1, 2))
        gh, gw = self.grid
        if vis.numel() != gh * gw:
            print("capture skipped: span", vis.numel(), "grid", self.grid); return
        row = attn_probs.float().mean(dim=(0, 1, 2)); K = int(row.numel()); al = int(action_len)
        self.rec.add_groups(layer_idx, row[:a].sum(), vis.sum(), row[K - al:].sum())
        self.rec.add(layer_idx, vis, num_views=1, grid=(gh, gw), view_major=True)


def compose_cfg(task, overrides):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str((REPO / "configs").resolve())):
        return compose(config_name="sim_robotwin.yaml", overrides=[f"task={task}"] + overrides)


def build_policy(args):
    if args.family == "flux2":
        overrides = [o.format(flux2_src=args.flux2_src, flux2_model=args.flux2_model, ae_model=args.ae_model, qwen3=args.qwen3_model) for o in FLUX2_OVERRIDES]
        layout = "compact_288x256"
    else:
        overrides = [o.format(qwen=args.qwen_model) for o in QWEN_OVERRIDES]
        layout = "qwen_horizontal_224"
    cfg = compose_cfg(args.task, overrides)
    usr = dict(sim_cfg_name="sim_robotwin.yaml", sim_task=args.task, ckpt_setting=str(args.ckpt), dataset_stats_path=str(args.stats),
               device="cuda", mixed_precision="bf16", action_horizon=16, replan_steps=16, num_inference_steps=args.steps, seed=args.seed,
               text_cfg_scale=1.0, rand_device="cpu", robotwin_camera_layout=layout, cache_text_context=False,
               model_overrides=OmegaConf.to_container(cfg.model, resolve=True))
    return dp.get_model(usr), layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["flux2", "qwen"])
    ap.add_argument("--task", required=True, help="Hydra task config name, e.g. robotwin_flux2_klein_4b_base_clean_jam")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flux2-src", default=str(REPO / "third_party/flux2"))
    ap.add_argument("--flux2-model", default=str(REPO / "checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"))
    ap.add_argument("--ae-model", default=str(REPO / "checkpoints/flux2/FLUX.2-dev/ae.safetensors"))
    ap.add_argument("--qwen3-model", default="/path/to/user/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
    ap.add_argument("--qwen-model", default=str(REPO / "checkpoints/Qwen3.5-2B-Base"))
    args = ap.parse_args()
    policy, layout = build_policy(args)
    model = policy.model
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records = json.load(open(args.manifest))
    for rec_in in records:
        rec = Recorder()
        obs = {"observation": {k + "_camera": {"rgb": np.asarray(Image.open(rec_in[k]).convert("RGB"))} for k in ("head", "left", "right")},
               "joint_action": {"vector": np.asarray(rec_in["state"], dtype=np.float32)}}
        if args.family == "flux2":
            model.mot.action_attention_capture = GridCapture(rec)
            undo = lambda: setattr(model.mot, "action_attention_capture", None)
        else:
            undo = patch_qwen(model, rec, grid=(7, 7), num_views=3)
        with torch.inference_mode():
            action = policy._infer_action_chunk(obs, rec_in["instruction"])
        undo()
        if not rec.maps:
            raise RuntimeError("no attention captured")
        layers = sorted({k[0] for k in rec.maps}); steps = sorted({k[1] for k in rec.maps})
        stack = np.stack([[rec.maps[(l, s)] for s in steps] for l in layers])
        gmass = np.stack([[rec.groups.get((l, s), np.full(4, np.nan, dtype=np.float32)) for s in steps] for l in layers])
        model_input = policy._build_robotwin_image_tensor(obs)[0].float().cpu()
        model_rgb = ((model_input.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
        np.savez_compressed(out / f"{rec_in['tag']}.npz", maps=stack, layers=np.array(layers), steps=np.array(steps),
                            head=np.asarray(Image.open(rec_in["head"]).convert("RGB")), model_rgb=model_rgb, layout=layout,
                            instruction=rec_in["instruction"], family=args.family, action=np.asarray(action),
                            meta=json.dumps(rec.meta), group_mass=gmass, group_names=np.array(["text", "visual", "action", "other"]),
                            record=json.dumps({k: v for k, v in rec_in.items() if k != "state"}))
        print(f"[{args.family}] {rec_in['tag']}: layers={len(layers)} steps={len(steps)} grid={stack.shape[-3:]} "
              f"visual share={np.nanmean(gmass[..., 1]):.3f} max={stack.mean(axis=(0, 1)).max():.4f}", flush=True)


if __name__ == "__main__":
    main()
