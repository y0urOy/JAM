"""Bit-level equivalence test: first_frame_only pipeline vs full pipeline (LIBERO).

Run on one GPU:
  python scripts/test_first_frame_equiv.py task=libero_bplus_na_downpred118_lam06_single18_2cam224_1e-4

  T1 dataset: actions/proprio/context/9-frame pad grid identical; pixels: new == old[:, :1]
  T2 VAE causality: latent frame 0 from 1-frame encode vs 9-frame encode
  T3 loss: training_loss with the SAME precomputed targets (target-cache path) on
     both pipelines — incl. episode-TAIL samples where offsets 1/2 are masked.
"""
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jam_wan.models.wan22.foresight import encode_foresight_hidden  # noqa: E402
from jam_wan.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    ds_full = instantiate(cfg.data.train)
    ds_ff = instantiate(cfg.data.train, first_frame_only=True)
    n = len(ds_full)

    normal_idx, pad_idx, i = [], [], n // 3
    while (len(pad_idx) < 2 or len(normal_idx) < 2) and i < n:
        has_pad = bool(ds_full._get(i)["image_is_pad"].any())
        if has_pad and len(pad_idx) < 2:
            pad_idx.append(i)
        elif not has_pad and len(normal_idx) < 2:
            normal_idx.append(i)
        i += 997
    picks = normal_idx[:2] + pad_idx[:2]
    print(f"test idx: normal={normal_idx[:2]} padded_tail={pad_idx[:2]}", flush=True)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["foresight"].pop("target_cache_roots", None)  # build WITH encoder (targets computed online here)
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    model.eval()
    model.foresight_use_target_cache = True  # flag-only: allow 1-frame video path + precomputed-loss branch
    layer = int(model.foresight_encoder_layers[0])

    all_ok = True
    for idx in picks:
        a, b = ds_full._get(idx), ds_ff._get(idx)

        # ---- T1: everything except pixels identical; pixels = frame 0 slice ----
        assert torch.equal(a["action"], b["action"]) and torch.equal(a["proprio"], b["proprio"])
        assert torch.equal(a["context"], b["context"])
        assert torch.equal(a["image_is_pad"], b["image_is_pad"]), "pad grid changed!"
        # _get returns video as [C, T, H, W] — slice the TIME dim
        assert a["video"].ndim == 4 and b["video"].ndim == 4, (a["video"].shape, b["video"].shape)
        va = a["video"][:, :1]
        assert va.shape == b["video"].shape, f"T1 shape mismatch @ {idx}: {va.shape} vs {b['video'].shape}"
        assert torch.equal(va.contiguous(), b["video"].contiguous()), f"T1 pixel mismatch @ {idx}"

        ba = torch.utils.data.default_collate([a])
        bb = torch.utils.data.default_collate([b])

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ia = model.build_inputs(ba)
            ib = model.build_inputs(bb)

            # ---- T2: VAE first-latent causality ----
            la0, lb0 = ia["input_latents"][:, :, 0:1].float(), ib["input_latents"][:, :, 0:1].float()
            bitwise = torch.equal(la0, lb0)
            rel = ((la0 - lb0).abs().max() / la0.abs().max().clamp(min=1e-8)).item()
            print(f"idx {idx}: T2 latent0 bitwise={bitwise} max_rel={rel:.2e}", flush=True)
            all_ok &= bitwise or rel < 3e-3

            # ---- T3: same precomputed target -> training_loss on both pipelines ----
            enc_h, _ = encode_foresight_hidden(
                encoder=model.foresight_encoder, clean_latents=ia["input_latents"],
                context=ia["context"], context_mask=ia["context_mask"], capture_layer=layer,
                fuse_vae_embedding_in_latents=ia["fuse_vae_embedding_in_latents"])
            cache = enc_h.to(torch.bfloat16).unsqueeze(1)  # [B, 1, S, D] as the cache wrapper would
            losses = {}
            for tag, batch in (("full", ba), ("ff", bb)):
                batch = dict(batch)
                batch["foresight_enc_cache"] = cache
                torch.manual_seed(1234)          # identical noise/timestep draws for both calls
                torch.cuda.manual_seed_all(1234)
                _, ld = model.training_loss(batch)
                losses[tag] = ld
            df = abs(losses["full"]["loss_foresight"] - losses["ff"]["loss_foresight"])
            da = abs(losses["full"]["loss_action"] - losses["ff"]["loss_action"])
            print(f"idx {idx}: T3 loss_foresight {losses['full']['loss_foresight']:.6f} vs "
                  f"{losses['ff']['loss_foresight']:.6f} (d={df:.2e}) | loss_action d={da:.2e}", flush=True)
            all_ok &= df < 1e-4 and da < 1e-4

    print("\nRESULT:", "EQUIVALENT (PASS)" if all_ok else "MISMATCH (FAIL)")


if __name__ == "__main__":
    main()
