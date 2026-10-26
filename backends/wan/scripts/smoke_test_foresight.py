"""CPU smoke test for the foresight components (tiny modules, no pretrained weights).

Run: python scripts/smoke_test_foresight.py
"""

import torch

from jam_wan.models.wan22.foresight import (
    SharedForesightProjector,
    encode_foresight_vae_direct,
    lift_foresight_vae_patches,
    patch_foresight_vae_latents,
)
from jam_wan.models.wan22.jam_wan import JAMWan
from jam_wan.models.wan22.mot import MoT
from jam_wan.models.wan22.wan_video_dit import WanVideoDiT


# attn_head_dim must keep the 3D-RoPE split even: head_dim//3 and head_dim-2*(head_dim//3)
# both even (true for the real model's 128; 24 is the smallest convenient choice here).
def tiny_video_dit(num_layers: int = 4, mask_mode: str = "first_frame_causal") -> WanVideoDiT:
    return WanVideoDiT(**tiny_video_dit_config(num_layers=num_layers, mask_mode=mask_mode))


def tiny_video_dit_config(num_layers: int = 4, mask_mode: str = "first_frame_causal") -> dict:
    return dict(
        hidden_dim=96,
        in_dim=48,
        ffn_dim=192,
        out_dim=48,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        attn_head_dim=24,
        num_layers=num_layers,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
        video_attention_mask_mode=mask_mode,
    )


def test_projector():
    proj = SharedForesightProjector(
        in_dim=64, out_dim=96, keys=["2_1", "2_2"], hidden_dim=32, num_layers=2, num_heads=4, ffn_dim=128
    )
    x = torch.randn(2, 16, 64)
    feat = proj.backbone(x)
    outs = [proj.head(feat, key) for key in ("2_1", "2_2")]
    assert feat.shape == (2, 16, 32), feat.shape
    assert all(out.shape == (2, 16, 96) for out in outs), [out.shape for out in outs]
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"[ok] shared projector forward: {tuple(x.shape)} -> 2 x {tuple(outs[0].shape)} ({n_params/1e3:.1f}K params)")


def test_vae_direct_target():
    """VAE patches must preserve token order and cosine geometry exactly."""
    latents = torch.randn(2, 48, 3, 8, 8, requires_grad=True)
    target, tokens_per_frame = encode_foresight_vae_direct(
        latents,
        patch_size=(1, 2, 2),
        output_dim=768,
    )
    assert target.shape == (2, 3 * 16, 768), target.shape
    assert tokens_per_frame == 16
    assert not target.requires_grad

    first_raw = latents.detach()[0, :, 0, :2, :2].reshape(-1)
    expected_first = first_raw.repeat(4) / 2.0
    assert torch.equal(target[0, 0], expected_first)

    # Confirm every raw patch/token follows Conv3d's exact flattening order.
    raw_tokens, _ = encode_foresight_vae_direct(
        latents.detach(),
        patch_size=(1, 2, 2),
        output_dim=192,
    )
    compact_tokens, compact_tpf = patch_foresight_vae_latents(
        latents.detach(),
        patch_size=(1, 2, 2),
    )
    assert compact_tpf == tokens_per_frame
    assert torch.equal(compact_tokens, raw_tokens)
    assert torch.equal(
        lift_foresight_vae_patches(compact_tokens, output_dim=768),
        target,
    )
    conv = torch.nn.Conv3d(48, 5, kernel_size=(1, 2, 2), stride=(1, 2, 2))
    conv_tokens = conv(latents.detach()).permute(0, 2, 3, 4, 1).reshape(2, 48, 5)
    manual_conv_tokens = torch.nn.functional.linear(
        raw_tokens, conv.weight.reshape(5, 192), conv.bias
    )
    assert torch.allclose(conv_tokens, manual_conv_tokens, atol=1e-6, rtol=1e-6)

    raw = torch.randn(2, 192)
    lifted, _ = encode_foresight_vae_direct(
        raw.reshape(2, 48, 1, 2, 2),
        patch_size=(1, 2, 2),
        output_dim=3072,
    )
    lifted = lifted[:, 0]
    assert torch.allclose(lifted.norm(dim=-1), raw.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(lifted[0] @ lifted[1], raw[0] @ raw[1], atol=2e-5, rtol=1e-5)
    print("[ok] pretrained VAE-direct target: 192->3072 fixed isometric lift")


def test_mot_hidden_capture():
    num_layers = 4
    video_expert = tiny_video_dit(num_layers=num_layers)
    action_expert = tiny_video_dit(num_layers=num_layers)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )

    batch, video_seq, action_seq, dim = 2, 12, 6, 96
    rope_half_dim = 24 // 2

    def rand_freqs(seq):
        return torch.polar(torch.ones(seq, 1, rope_half_dim), torch.randn(seq, 1, rope_half_dim))

    total = video_seq + action_seq
    kwargs = dict(
        embeds_all={
            "video": torch.randn(batch, video_seq, dim),
            "action": torch.randn(batch, action_seq, dim),
        },
        attention_mask=torch.ones(total, total, dtype=torch.bool),
        freqs_all={"video": rand_freqs(video_seq), "action": rand_freqs(action_seq)},
        context_all={"video": None, "action": None},
        t_mod_all={
            "video": torch.randn(batch, 6, dim),
            "action": torch.randn(batch, 6, dim),
        },
    )

    torch.manual_seed(0)
    tokens_plain = mot(**kwargs)
    torch.manual_seed(0)
    tokens_out, hidden = mot(**kwargs, return_video_hidden_layers=[2, num_layers])

    assert set(hidden.keys()) == {2, num_layers}
    assert hidden[2].shape == (batch, video_seq, dim)
    # capture at final depth equals the returned video tokens
    assert torch.equal(hidden[num_layers], tokens_out["video"])
    # capture must not perturb the forward computation
    assert torch.equal(tokens_plain["video"], tokens_out["video"])
    assert torch.equal(tokens_plain["action"], tokens_out["action"])
    # mid-depth capture differs from the final output (sanity)
    assert not torch.allclose(hidden[2], tokens_out["video"])
    print(f"[ok] MoT hidden capture: layers {sorted(hidden.keys())}, shape {tuple(hidden[2].shape)}")


def test_gradient_flow():
    """Foresight loss must backprop into projector AND the student video tokens."""
    proj = SharedForesightProjector(
        in_dim=64, out_dim=64, keys=["1_1"], num_layers=2, num_heads=4, ffn_dim=128
    )
    h_obs = torch.randn(2, 16, 64, requires_grad=True)
    target = torch.randn(2, 16, 64)
    pred = proj.head(proj.backbone(h_obs), "1_1")
    cos = torch.nn.functional.cosine_similarity(pred.float(), target.float(), dim=-1)
    loss = (1.0 - cos.mean(dim=1)).mean()
    loss.backward()
    assert h_obs.grad is not None and h_obs.grad.abs().sum() > 0
    assert all(p.grad is not None for p in proj.parameters() if p.requires_grad)
    print(f"[ok] gradient flow: loss={loss.item():.4f}, |dL/dh|={h_obs.grad.abs().mean().item():.6f}")


class TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4


def make_tiny_jam_wan(**foresight_overrides) -> JAMWan:
    config = tiny_video_dit_config(num_layers=4)
    config.update(hidden_dim=192, ffn_dim=384, num_heads=8)
    video = WanVideoDiT(**config)
    action = WanVideoDiT(**config)
    foresight_cfg = {
        "enabled": True,
        "video_cotrain": False,
        "target_type": "vae_direct",
        "teacher_mode": "frozen_pretrained",
        "layers": [2],
        "encoder_layers": [0],
        "offsets": [1, 2],
        "per_frame": True,
        "shared_projector": True,
        "projector_hidden_dim": 96,
        "projector_num_layers": 1,
        "projector_num_heads": 4,
        "projector_ffn_dim": 192,
        "action_layers": None,
    }
    foresight_cfg.update(foresight_overrides)
    return JAMWan(
        video_expert=video,
        action_expert=action,
        mot=MoT(
            mixtures={"video": video, "action": action},
            mot_checkpoint_mixed_attn=False,
        ),
        vae=TinyVAE(),
        text_dim=32,
        device="cpu",
        torch_dtype=torch.float32,
        foresight_cfg=foresight_cfg,
    )


def test_jam_wan_vae_direct_loss():
    """Exercise the VAE-direct target through JAMWan's real loss method."""
    model = make_tiny_jam_wan()
    latents = torch.randn(2, 48, 3, 8, 8)
    student = torch.randn(2, 16, 192, requires_grad=True)
    loss = model._compute_foresight_loss(
        video_hidden_by_layer={2: student},
        tokens_per_frame=16,
        input_latents=latents,
        image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    print(f"[ok] JAMWan VAE-direct loss + gradient: loss={loss.item():.4f}")


def test_cached_vae_direct_loss_equivalence():
    """Compact raw VAE cache must reproduce the online VAE-direct loss."""
    model = make_tiny_jam_wan(target_cache_roots=["synthetic-compact-cache"])
    latents = torch.randn(2, 48, 3, 8, 8)
    raw, tpf = patch_foresight_vae_latents(latents, patch_size=(1, 2, 2))
    student = torch.randn(2, tpf, 192)
    kwargs = dict(
        video_hidden_by_layer={2: student},
        tokens_per_frame=tpf,
        input_latents=latents,
        image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
    )
    online = model._compute_foresight_loss(**kwargs)
    cached = model._compute_foresight_loss(
        **kwargs,
        precomputed_enc_cache={0: raw},
    )
    assert torch.equal(online, cached), (online, cached)
    print(f"[ok] compact cached VAE-direct == online loss exactly: {online.item():.6f}")


def test_only_released_objective_is_accepted():
    """Settings outside the released VAE-direct objective fail at construction."""
    for override in (
        {"target_type": "encoder_hidden"},
        {"teacher_mode": "frozen_random"},
        {"encoder_layers": [2]},
        {"per_frame": False},
        {"shared_projector": False},
        {"projector_type": "mlp"},
        {"action_layers": [2]},
    ):
        try:
            make_tiny_jam_wan(**override)
        except ValueError:
            continue
        raise AssertionError(f"expected a ValueError for {override}")
    print("[ok] only the released VAE-direct foresight objective is accepted")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_projector()
    test_vae_direct_target()
    test_mot_hidden_capture()
    test_gradient_flow()
    test_jam_wan_vae_direct_loss()
    test_cached_vae_direct_loss_equivalence()
    test_only_released_objective_is_accepted()
    print("\nAll foresight smoke tests passed.")
