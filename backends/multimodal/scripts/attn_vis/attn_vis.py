"""Action-to-visual attention maps for JAM policies on LIBERO frames.

For one trained policy (a run directory with config.yaml, dataset_stats.json and a
weights checkpoint) and a manifest of (agent image, wrist image, state, instruction)
records, run the action inference loop while recording, at every action-expert
attention layer, how much the action tokens attend to each visual token of the
current observation.  Maps are averaged over heads and action queries, stored per
layer and denoising step, and saved with the reshaped per-view grids.

Families:
  yak       Ovis-U1 Yak18 + ActionDiTYak18 (MoT capture protocol)
  dino      DINOv2-B/L + native ActionDINO (SDPA recomputed from q/k)
  videomae  VideoMAE-B + native ActionVideoMAE12 (same path as dino)
  qwen      Qwen3.5-2B + causal ActionQwen24 (full-attention layers only)
"""
import argparse, json, os, sys, types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from jam_multimodal.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from jam_multimodal.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


class ev:
    """Inlined copies of the helpers in experiments/libero/eval_libero_single.py.

    Importing that module pulls in LIBERO/robosuite (simulator, EGL, /tmp logs),
    none of which is needed to run a policy on stored frames.
    """
    import importlib as _il, inspect as _insp, logging as _log

    @staticmethod
    def _is_none_like(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "none", "null"}
        return False

    @staticmethod
    def _resolve_target(target):
        module_name, _, attr_name = target.rpartition(".")
        return getattr(ev._il.import_module(module_name), attr_name)

    @staticmethod
    def _filter_model_cfg_for_target(model_cfg):
        model_dict = OmegaConf.to_container(model_cfg, resolve=True)
        target = model_dict.get("_target_")
        if ev._is_none_like(target):
            return OmegaConf.create(model_dict)
        sig = ev._insp.signature(ev._resolve_target(str(target)))
        if any(p.kind == ev._insp.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return OmegaConf.create(model_dict)
        allowed = {"_target_"} | {n for n, p in sig.parameters.items()
                                   if p.kind in {ev._insp.Parameter.POSITIONAL_OR_KEYWORD, ev._insp.Parameter.KEYWORD_ONLY}}
        dropped = sorted(set(model_dict) - allowed)
        if dropped:
            print("ignoring unsupported model config keys:", dropped)
        return OmegaConf.create({k: v for k, v in model_dict.items() if k in allowed})

    @staticmethod
    def _resolve_model_cfg(cfg):
        cfg_r = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False, throw_on_missing=False))
        model_cfg = cfg_r.model
        path = cfg.get("model_overrides_path", None)
        if not ev._is_none_like(path):
            model_cfg = OmegaConf.merge(model_cfg, OmegaConf.load(str(path)))
        inline = cfg.get("model_overrides", None)
        if not ev._is_none_like(inline):
            model_cfg = OmegaConf.merge(model_cfg, inline)
        cfg_r.model = model_cfg
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg_r.model, resolve=True))
        return ev._filter_model_cfg_for_target(model_cfg)

    @staticmethod
    def _disable_training_only_foresight_for_eval(model_cfg, cfg):
        fc = model_cfg.get("foresight", None)
        if fc is not None and bool(fc.get("enabled", False)):
            model_cfg.foresight.enabled = False

    @staticmethod
    def _mixed_precision_to_model_dtype(mp):
        key = str(mp).strip().lower()
        return {"no": torch.float32, "fp16": torch.float16}.get(key, torch.bfloat16)

    @staticmethod
    def _center_crop_resize(image, width, height):
        pil_image = Image.fromarray(image)
        src_w, src_h = pil_image.size
        scale = max(width / src_w, height / src_h)
        resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0); top = max((rh - height) // 2, 0)
        return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)

    @staticmethod
    def _normalize_proprio(proprio, processor):
        state_key = processor.shape_meta["state"][0]["key"]
        batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
        batch = processor.action_state_transform(batch)
        batch = processor.normalizer.forward(batch)
        return batch["state"][state_key]

    @staticmethod
    def _load_model_checkpoint(model, ckpt, mmap_checkpoint=False):
        kw = {}
        if mmap_checkpoint and "mmap" in ev._insp.signature(model.load_checkpoint).parameters:
            kw["mmap"] = True
        model.load_checkpoint(ckpt, **kw)


# ----------------------------------------------------------------------------- capture
class Recorder:
    """Collects [layer][step] -> np.ndarray [V, gh, gw] attention mass per visual token."""

    def __init__(self):
        self.maps = {}      # (layer, step) -> np.ndarray [V, gh, gw]
        self.step = 0
        self.count = {}     # layer -> number of times seen (one per denoising step)
        self.meta = {}
        self.groups = {}    # (layer, step) -> np.ndarray [4]: mass on text, visual, action, other keys
        self.gcount = {}

    def add_groups(self, layer, text, visual, action):
        # scalars: attention mass (already averaged over batch/heads/queries) on each key group
        vals = [float(text), float(visual), float(action)]
        vals.append(max(0.0, 1.0 - sum(vals)))
        layer = int(layer); step = self.gcount.get(layer, 0); self.gcount[layer] = step + 1
        self.groups[(layer, step)] = np.array(vals, dtype=np.float32)

    def add(self, layer, probs_over_visual, num_views, grid, view_major=True):
        # probs_over_visual: [K_visual] float (already averaged over batch/heads/queries)
        arr = probs_over_visual.detach().float().cpu().numpy()
        gh, gw = grid
        if view_major:
            arr = arr.reshape(num_views, gh, gw)
        else:  # single wide grid with views side by side along width
            wide = arr.reshape(gh, gw)
            arr = np.stack(np.split(wide, num_views, axis=1), axis=0)
        layer = int(layer)
        step = self.count.get(layer, 0)
        self.count[layer] = step + 1
        self.maps[(layer, step)] = arr


class MoTCapture:
    """Implements the duck-typed `mot.action_attention_capture` protocol."""

    def __init__(self, rec: Recorder):
        self.rec = rec
        self.cond = None
        self.grid = None
        self._seen_first = False

    def configure(self, condition_slice, condition_grid, prefix_len, action_len, metadata):
        self.cond = condition_slice
        self.grid = tuple(int(x) for x in condition_grid)
        self.rec.meta = {"condition_slice": list(condition_slice), "grid": list(self.grid),
                         "prefix_len": int(prefix_len), "action_len": int(action_len), **(metadata or {})}

    def start_step(self, step_idx):
        self.rec.step = int(step_idx)

    def should_capture(self, layer_idx):
        return True

    def update(self, attn_probs, *, layer_idx, block_type, prefix_len, action_len):
        # attn_probs: [B, H, Q(action), K]
        if self.cond is None:
            raise RuntimeError("MoT capture not configured (Ovis-U1 path needs manual configure).")
        a, b = int(self.cond[0]), int(self.cond[1])
        vis = attn_probs[:, :, :, a:b].float().mean(dim=(0, 1, 2))  # [ref_len]
        gh, gw = self.grid
        if vis.numel() != gh * gw:
            print("capture skipped: span", vis.numel(), "grid", self.grid); return
        row = attn_probs.float().mean(dim=(0, 1, 2))                 # [K] mass per key
        K = int(row.numel()); al = int(action_len)
        text = row[:a].sum() if K == int(prefix_len) + al else row.new_zeros(())   # image-self calls carry no text keys
        self.rec.add_groups(layer_idx, text, vis.sum(), row[K - al:].sum())
        # Yak/Ovis packs the 224x448 latent as one 14x28 grid: cols 0-13 agent, 14-27 wrist.
        self.rec.add(layer_idx, vis, num_views=2, grid=(gh, gw), view_major=False)


def patch_dino_sdpa(model, rec: Recorder, num_views, patches_per_view, grid):
    """Wrap SDPA inside action_dit_dino to recompute softmax(QK^T) for action queries."""
    import jam_multimodal.models.backbones.action_dit_dino as adm
    orig = adm.F.scaled_dot_product_attention
    state = {"layer": 0}
    expert = model.action_expert
    orig_layer = expert.forward_action_layer

    def forward_action_layer(hidden_states, timestep_embedding, *, layer_index, condition):
        state["layer"] = int(layer_index)
        return orig_layer(hidden_states, timestep_embedding, layer_index=layer_index, condition=condition)

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        out = orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        try:
            B, H, Q, D = q.shape
            K = k.shape[2]
            nvis = num_views * patches_per_view
            if K > nvis and Q <= 64:  # action-layer call: [text | visual | action]
                s = scale if scale is not None else D ** -0.5
                logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * s
                if attn_mask is not None:
                    logits = logits.masked_fill(~attn_mask.to(torch.bool), torch.finfo(logits.dtype).min)
                probs = torch.softmax(logits, dim=-1)
                text_len = K - nvis - Q
                vis = probs[:, :, :, text_len:text_len + nvis].mean(dim=(0, 1, 2))
                rec.add(state["layer"], vis, num_views=num_views, grid=grid, view_major=True)
                row = probs.mean(dim=(0, 1, 2))
                rec.add_groups(state["layer"], row[:text_len].sum(), vis.sum(), row[text_len + nvis:].sum())
        except Exception as exc:  # pragma: no cover
            print("capture skipped:", exc)
        return out

    adm.F.scaled_dot_product_attention = sdpa
    expert.forward_action_layer = forward_action_layer
    return lambda: setattr(adm.F, "scaled_dot_product_attention", orig)


def patch_qwen(model, rec: Recorder, grid=(7, 7), num_views=2):
    """Record image-token positions and full-attention weights of ActionQwen24."""
    import jam_multimodal.models.backbones.jam_qwen35 as qm
    import jam_multimodal.models.backbones.action_dit_qwen35 as aq
    holder = {"image_idx": None}
    orig_extract = qm.extract_grouped_visual_tokens

    def extract(hidden_states, input_ids, *, image_token_id, num_images_per_sample,
                visual_tokens_per_image, token_indices=None):
        if token_indices is not None:
            holder["image_idx"] = token_indices[0].detach().cpu()
        else:
            holder["image_idx"] = torch.nonzero(input_ids[0].eq(int(image_token_id))).flatten().cpu()
        return orig_extract(hidden_states, input_ids, image_token_id=image_token_id,
                            num_images_per_sample=num_images_per_sample,
                            visual_tokens_per_image=visual_tokens_per_image, token_indices=token_indices)

    qm.extract_grouped_visual_tokens = extract
    # Second route: the native path pops `_image_token_indices` from the prepared inputs.
    orig_prepare = model._prepare_qwen_inputs

    def prepare(*a, **kw):
        out = orig_prepare(*a, **kw)
        idx = out.get("_image_token_indices", None) if isinstance(out, dict) else None
        if idx is not None:
            holder["image_idx"] = idx[0].detach().reshape(-1).cpu()
        elif isinstance(out, dict) and "input_ids" in out:
            holder["image_idx"] = torch.nonzero(out["input_ids"][0].eq(int(model.image_token_id))).flatten().cpu()
        return out

    model._prepare_qwen_inputs = prepare

    expert = model.action_expert
    state = {"layer": 0}
    orig_block = expert._run_block_action_suffix

    def run_block(block, packed, **kw):
        state["layer"] = int(getattr(block, "_attn_vis_index", -1))
        return orig_block(block, packed, **kw)

    for i, blk in enumerate(expert.blocks):
        blk._attn_vis_index = i
    expert._run_block_action_suffix = run_block

    def full_attn(self, mixer, hidden_states, *, position_embeddings, attention_mask, action_indices):
        batch_size, seq_len, _ = hidden_states.shape
        action_len = int(action_indices.shape[1])
        action_input = self._gather_action(hidden_states, action_indices)
        action_shape = action_input.shape[:-1]
        query_states, gate = torch.chunk(mixer.q_proj(action_input).view(*action_shape, -1, mixer.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*action_shape, -1)
        query_states = mixer.q_norm(query_states.view(*action_shape, -1, mixer.head_dim)).transpose(1, 2)
        key_states = mixer.k_norm(mixer.k_proj(hidden_states).view(batch_size, seq_len, -1, mixer.head_dim)).transpose(1, 2)
        value_states = mixer.v_proj(hidden_states).view(batch_size, seq_len, -1, mixer.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        action_cos = cos.gather(1, action_indices[:, :, None].expand(-1, -1, cos.shape[-1]))
        action_sin = sin.gather(1, action_indices[:, :, None].expand(-1, -1, sin.shape[-1]))
        query_states = self._rotate_single(query_states, action_cos, action_sin)
        key_states = self._rotate_single(key_states, cos, sin)
        if mixer.num_key_value_groups > 1:
            key_states = key_states[:, :, None].expand(batch_size, key_states.shape[1], mixer.num_key_value_groups, seq_len, mixer.head_dim).reshape(batch_size, -1, seq_len, mixer.head_dim)
            value_states = value_states[:, :, None].expand(batch_size, value_states.shape[1], mixer.num_key_value_groups, seq_len, mixer.head_dim).reshape(batch_size, -1, seq_len, mixer.head_dim)
        if int(attention_mask.shape[-2]) == action_len:
            action_mask = attention_mask
        else:
            action_mask = attention_mask.gather(2, action_indices[:, None, :, None].expand(-1, attention_mask.shape[1], -1, seq_len))
        w = torch.matmul(query_states, key_states.transpose(2, 3)) * mixer.scaling + action_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32)
        idx = holder["image_idx"]
        if idx is not None:
            vis = w[:, :, :, idx.to(w.device)].mean(dim=(0, 1, 2))  # [98]
            rec.add(state["layer"], vis, num_views=num_views, grid=grid, view_major=True)
            row = w.mean(dim=(0, 1, 2))                              # [seq_len]
            act = row[action_indices[0].to(w.device)].sum()
            rec.add_groups(state["layer"], row.sum() - vis.sum() - act, vis.sum(), act)
        out = torch.matmul(w.to(query_states.dtype), value_states).transpose(1, 2).contiguous()
        out = out.reshape(batch_size, action_len, -1) * torch.sigmoid(gate)
        return mixer.o_proj(out)

    expert._run_full_attention_action_suffix = types.MethodType(full_attn, expert)
    return lambda: None


# ----------------------------------------------------------------------------- main
def build(run_dir: Path, ckpt: Path, device: str):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.ckpt = str(ckpt)
    if "EVALUATION" not in cfg:
        cfg.EVALUATION = {}
    cfg.EVALUATION.disable_foresight = True
    cfg.EVALUATION.dataset_stats_path = str(run_dir / "dataset_stats.json")
    dtype = ev._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model_cfg = ev._resolve_model_cfg(cfg)
    ev._disable_training_only_foresight_for_eval(model_cfg, cfg)
    model = instantiate(model_cfg, model_dtype=dtype, device=device)
    ev._load_model_checkpoint(model, str(ckpt), mmap_checkpoint=True)
    model = model.to(device).eval()
    stats = load_dataset_stats_from_json(str(run_dir / "dataset_stats.json"))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    return cfg, model, processor, dtype


def make_input(rec, cfg, processor, device, dtype):
    meta = processor.shape_meta["images"]
    h0, w0 = int(meta[0]["shape"][1]), int(meta[0]["shape"][2])
    h1, w1 = int(meta[1]["shape"][1]), int(meta[1]["shape"][2])
    agent = ev._center_crop_resize(np.asarray(Image.open(rec["image"]).convert("RGB")), width=w0, height=h0)
    wrist = ev._center_crop_resize(np.asarray(Image.open(rec["wrist_image"]).convert("RGB")), width=w1, height=h1)
    rgb = np.concatenate([agent, wrist], axis=1)
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype) * (2.0 / 255.0) - 1.0
    proprio = ev._normalize_proprio(np.asarray(rec["state"], dtype=np.float32), processor)
    return x, proprio, agent, wrist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--family", required=True, choices=["yak", "dino", "videomae", "qwen"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_dir = Path(args.run)
    ckpt = Path(args.ckpt) if args.ckpt else run_dir / "checkpoints" / "weights" / "step_021700.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, processor, dtype = build(run_dir, ckpt, device)
    action_horizon = int(cfg.EVALUATION.get("action_horizon", None) or int(cfg.data.train.num_frames) - 1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records = json.load(open(args.manifest))

    for rec_in in records:
        rec = Recorder()
        x, proprio, agent, wrist = make_input(rec_in, cfg, processor, device, dtype)
        if args.family == "yak":
            cap = MoTCapture(rec)
            model.mot.action_attention_capture = cap
            # The Ovis-U1 inference path never calls configure(); derive the
            # condition-token span from the mask builder's arguments instead.
            orig_mask = model._build_mot_attention_mask_ovis_u1

            def build_mask(*a, **kw):
                txt, tgt, cond = int(kw["txt_len"]), int(kw["target_len"]), int(kw["cond_len"])
                gh, gw = int(x.shape[-2]) // 16, int(x.shape[-1]) // 16
                if cond != gh * gw:
                    raise RuntimeError(f"cond_len {cond} != grid {gh}x{gw}")
                cap.configure(condition_slice=(txt + tgt, txt + tgt + cond), condition_grid=(gh, gw),
                              prefix_len=txt + tgt + cond, action_len=int(kw.get("action_len", 0)),
                              metadata={"txt_len": txt, "target_len": tgt, "cond_len": cond, "source": "ovis_u1"})
                return orig_mask(*a, **kw)

            model._build_mot_attention_mask_ovis_u1 = build_mask
            # `forward_yak_action_with_direct_prefix_cache` never consults the capture
            # object, so record probabilities directly from the mixed-attention kernel.
            orig_mixed = model.mot._mixed_attention
            n_double = int(len(getattr(model.action_expert, "double_blocks", [])))
            counts = {"joint": 0, "self": 0}

            def mixed(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=False):
                # Yak direct-prefix loop: double blocks run a joint call over
                # [txt | img | action] and an image-self call over [img | action];
                # single blocks run one call over [txt | img | action].
                if int(q_cat.shape[1]) == action_horizon and cap.cond is not None:
                    K = int(k_cat.shape[1]); A = action_horizon
                    txt0 = int(cap.cond[0]); img = int(cap.grid[0] * cap.grid[1])
                    out, probs = orig_mixed(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=True)
                    if K == txt0 + img + A:          # joint (double) or single block
                        layer = counts["joint"]; counts["joint"] += 1
                        vis = probs[:, :, :, txt0:txt0 + img].float().mean(dim=(0, 1, 2))
                        rec.add(layer, vis, num_views=2, grid=cap.grid, view_major=False)
                        row = probs.float().mean(dim=(0, 1, 2))   # keys are [text | image | action]
                        rec.add_groups(layer, row[:txt0].sum(), vis.sum(), row[txt0 + img:].sum())
                    elif K == img + A:               # image-self attention of a double block
                        layer = counts["self"]; counts["self"] += 1
                        vis = probs[:, :, :, :img].float().mean(dim=(0, 1, 2))
                        rec.add(100 + layer, vis, num_views=2, grid=cap.grid, view_major=False)
                    else:
                        print("unexpected key length", K, "txt", txt0, "img", img)
                    # one denoising step = n_double joint + n_double self + single calls;
                    # per-layer counters in Recorder separate steps, so reset here when a step ends
                    if counts["joint"] and counts["self"] == n_double and counts["joint"] >= n_double and (counts["joint"] - n_double) == len(getattr(model.action_expert, "single_blocks", [])):
                        counts["joint"] = 0; counts["self"] = 0
                    return out
                return orig_mixed(q_cat, k_cat, v_cat, attention_mask, return_attn_probs=return_attn_probs)

            model.mot._mixed_attention = mixed

            def undo():
                model._build_mot_attention_mask_ovis_u1 = orig_mask
                model.mot._mixed_attention = orig_mixed
        elif args.family in ("dino", "videomae"):
            side = 16 if args.family == "dino" else 14
            undo = patch_dino_sdpa(model, rec, num_views=2, patches_per_view=side * side, grid=(side, side))
        else:
            undo = patch_qwen(model, rec)
        prompt = DEFAULT_PROMPT.format(task=rec_in["instruction"])
        with torch.inference_mode():
            pred = model.infer_action(prompt=prompt, input_image=x, action_horizon=action_horizon,
                                      proprio=proprio.to(device=device, dtype=dtype), num_inference_steps=args.steps,
                                      seed=args.seed, rand_device="cpu")
        undo()
        layers = sorted({k[0] for k in rec.maps}); steps = sorted({k[1] for k in rec.maps})
        if not rec.maps:
            raise RuntimeError("no attention captured")
        stack = np.stack([[rec.maps[(l, s)] for s in steps] for l in layers])  # [L, S, V, gh, gw]
        gmass = np.stack([[rec.groups.get((l, s), np.full(4, np.nan, dtype=np.float32)) for s in steps] for l in layers])  # [L, S, 4]
        tag = rec_in.get("tag") or f"{rec_in['suite']}_t{rec_in['task_index']}_e{rec_in['episode_index']}_f{rec_in['frame_index']}"
        np.savez_compressed(out / f"{tag}.npz", maps=stack, layers=np.array(layers), steps=np.array(steps),
                            agent=agent, wrist=wrist, instruction=rec_in["instruction"], family=args.family,
                            action=pred["action"].numpy(), meta=json.dumps(rec.meta),
                            group_mass=gmass, group_names=np.array(["text", "visual", "action", "other"]),
                            record=json.dumps({k: v for k, v in rec_in.items() if k not in ("state",)}))
        print(f"[{args.family}] {tag}: layers={len(layers)} steps={len(steps)} grid={stack.shape[-2:]} "
              f"mean-over-all max={stack.mean(axis=(0,1)).max():.4f}")


if __name__ == "__main__":
    main()
