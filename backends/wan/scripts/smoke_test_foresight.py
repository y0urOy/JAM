"""CPU smoke test for the foresight components (tiny modules, no pretrained weights).

Run: python scripts/smoke_test_foresight.py
"""

import torch

from jam_wan.models.wan22.foresight import (
    ForesightProjector,
    build_foresight_ema_shadow,
    build_foresight_encoder,
    build_random_foresight_encoder,
    build_random_foresight_vae,
    encode_foresight_patch_embedding,
    encode_foresight_hidden,
    encode_foresight_vae_direct,
    fuse_future_targets,
    lift_foresight_vae_patches,
    patch_foresight_vae_latents,
    update_foresight_ema_teacher_,
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
    proj = ForesightProjector(in_dim=64, out_dim=96, num_layers=3, num_heads=4, ffn_dim=128)
    x = torch.randn(2, 16, 64)
    out = proj(x)
    assert out.shape == (2, 16, 96), out.shape
    n_params = sum(p.numel() for p in proj.parameters())
    print(f"[ok] projector forward: {tuple(x.shape)} -> {tuple(out.shape)} ({n_params/1e3:.1f}K params)")


def test_encoder_and_fusion():
    student = tiny_video_dit()
    encoder = build_foresight_encoder(student)
    assert encoder.video_attention_mask_mode == "bidirectional"
    assert all(not p.requires_grad for p in encoder.parameters())
    # frozen copy must hold identical weights
    for (n1, p1), (n2, p2) in zip(student.named_parameters(), encoder.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1

    # An independently loaded pretrained teacher can be adopted directly,
    # avoiding another multi-billion-parameter deepcopy.
    independent_source = tiny_video_dit()
    adopted_encoder = build_foresight_encoder(
        independent_source,
        max_layers=2,
        copy_model=False,
        source="independent-pretrained-test",
    )
    assert adopted_encoder is independent_source
    assert len(adopted_encoder.blocks) == 2
    assert all(not p.requires_grad for p in adopted_encoder.parameters())

    batch, frames, height, width = 2, 3, 8, 8
    latents = torch.randn(batch, 48, frames, height, width)
    context = torch.randn(batch, 5, 32)
    context_mask = torch.ones(batch, 5, dtype=torch.bool)

    hidden, tokens_per_frame = encode_foresight_hidden(
        encoder=encoder,
        clean_latents=latents,
        context=context,
        context_mask=context_mask,
        capture_layer=2,
    )
    assert tokens_per_frame == (height // 2) * (width // 2), tokens_per_frame
    assert hidden.shape == (batch, frames * tokens_per_frame, 96), hidden.shape
    assert hidden.requires_grad is False

    # fusion: offsets=None averages all frames
    target_all, valid_all = fuse_future_targets(hidden, tokens_per_frame, None, None)
    manual = hidden.view(batch, frames, tokens_per_frame, 96).mean(dim=1)
    assert torch.allclose(target_all, manual, atol=1e-6)
    assert valid_all.all()

    # fusion with explicit offsets and a pad mask
    frame_valid = torch.tensor([[True, True, False], [True, False, False]])
    target, valid = fuse_future_targets(hidden, tokens_per_frame, [1, 2], frame_valid)
    h = hidden.view(batch, frames, tokens_per_frame, 96)
    assert torch.allclose(target[0], h[0, 1], atol=1e-6)  # frame 2 padded -> only frame 1
    assert bool(valid[0]) is True
    assert bool(valid[1]) is False  # frames 1 and 2 both padded
    print(f"[ok] encoder capture + target fusion: hidden {tuple(hidden.shape)}, tpf={tokens_per_frame}")


def test_teacher_variants_and_ema():
    student = tiny_video_dit(num_layers=4)

    # Random-teacher construction must be deterministic, independent of the
    # student, prefix-truncated, and invisible to the caller's RNG stream.
    torch.manual_seed(1234)
    expected_next = torch.rand(8)
    torch.manual_seed(1234)
    random_teacher = build_random_foresight_encoder(
        tiny_video_dit_config(num_layers=4),
        max_layers=2,
        seed=77,
        device="cpu",
        torch_dtype=torch.float32,
    )
    actual_next = torch.rand(8)
    assert torch.equal(actual_next, expected_next)
    assert len(random_teacher.blocks) == 2
    assert random_teacher.video_attention_mask_mode == "bidirectional"
    assert all(not parameter.requires_grad for parameter in random_teacher.parameters())
    assert not torch.equal(
        random_teacher.patch_embedding.weight,
        student.patch_embedding.weight,
    )

    random_teacher_2 = build_random_foresight_encoder(
        tiny_video_dit_config(num_layers=4),
        max_layers=2,
        seed=77,
        device="cpu",
        torch_dtype=torch.float32,
    )
    for (_, first), (_, second) in zip(
        random_teacher.named_parameters(), random_teacher_2.named_parameters()
    ):
        assert torch.equal(first, second)

    # Random Layer0 is the exact seed-matched patch embedding from the full
    # random teacher and executes no Wan blocks or context/timestep path.
    layer0_teacher = build_random_foresight_encoder(
        tiny_video_dit_config(num_layers=4),
        max_layers=0,
        seed=77,
        device="cpu",
        torch_dtype=torch.float32,
    )
    assert len(layer0_teacher.blocks) == 0
    assert torch.equal(
        layer0_teacher.patch_embedding.weight,
        random_teacher.patch_embedding.weight,
    )
    latents = torch.randn(2, 48, 3, 8, 8)
    context = torch.randn(2, 5, 32)
    context_mask = torch.ones(2, 5, dtype=torch.bool)
    layer0_hidden, layer0_tpf = encode_foresight_patch_embedding(
        layer0_teacher, latents
    )
    full_pre = random_teacher.pre_dit(
        x=latents,
        timestep=torch.zeros(2),
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    assert layer0_tpf == int(full_pre["meta"]["tokens_per_frame"])
    assert torch.equal(layer0_hidden, full_pre["tokens"])

    # Match training precision: bf16 student/teacher with a separate fp32
    # master, so small EMA increments are not lost to bf16 rounding.
    ema_student = tiny_video_dit(num_layers=4).to(torch.bfloat16)
    ema_teacher = build_foresight_encoder(ema_student, max_layers=2)
    shadow = build_foresight_ema_shadow(ema_teacher)
    teacher_before = {
        name: parameter.detach().float().clone()
        for name, parameter in ema_teacher.named_parameters()
    }
    with torch.no_grad():
        for parameter in ema_student.parameters():
            parameter.add_(0.25)
    update_foresight_ema_teacher_(
        ema_teacher,
        ema_student,
        shadow,
        decay=0.5,
    )
    student_parameters = dict(ema_student.named_parameters())
    for name, teacher_parameter in ema_teacher.named_parameters():
        expected = teacher_before[name].mul(0.5).add(
            student_parameters[name].float(), alpha=0.5
        )
        assert torch.equal(shadow[name], expected)
        assert torch.equal(teacher_parameter, expected.to(teacher_parameter.dtype))
        assert teacher_parameter.grad is None
    print("[ok] random frozen teacher + fp32 online EMA teacher")


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


def test_random_teacher_vae_builder():
    """Random teacher VAE must be deterministic, frozen, and RNG-isolated."""

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv3d(3, 8, kernel_size=3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(8, 4, kernel_size=1),
            )

    torch.manual_seed(7)
    reference = TinyVAE()
    state_before = torch.random.get_rng_state().clone()
    teacher_a = build_random_foresight_vae(
        reference, seed=42, device="cpu", torch_dtype=torch.float32
    )
    state_after = torch.random.get_rng_state().clone()
    teacher_b = build_random_foresight_vae(
        reference, seed=42, device="cpu", torch_dtype=torch.float32
    )

    assert torch.equal(state_before, state_after), "random teacher VAE shifted run RNG"
    assert not teacher_a.training
    assert all(not parameter.requires_grad for parameter in teacher_a.parameters())
    assert all(
        torch.equal(value, teacher_b.state_dict()[name])
        for name, value in teacher_a.state_dict().items()
    )
    assert any(
        not torch.equal(value, reference.state_dict()[name])
        for name, value in teacher_a.state_dict().items()
    )
    print("[ok] random frozen teacher VAE: deterministic + frozen + RNG-isolated")


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
    proj = ForesightProjector(in_dim=64, out_dim=64, num_layers=2, num_heads=4, ffn_dim=128)
    h_obs = torch.randn(2, 16, 64, requires_grad=True)
    target = torch.randn(2, 16, 64)
    cos = torch.nn.functional.cosine_similarity(proj(h_obs).float(), target.float(), dim=-1)
    loss = (1.0 - cos.mean(dim=1)).mean()
    loss.backward()
    assert h_obs.grad is not None and h_obs.grad.abs().sum() > 0
    assert all(p.grad is not None for p in proj.parameters() if p.requires_grad)
    print(f"[ok] gradient flow: loss={loss.item():.4f}, |dL/dh|={h_obs.grad.abs().mean().item():.6f}")


def test_jam_wan_target_dispatch():
    """Exercise both new target branches through JAMWan's real loss method."""

    class TinyVAE(torch.nn.Module):
        temporal_downsample_factor = 4

    config = tiny_video_dit_config(num_layers=4)
    config.update(hidden_dim=192, ffn_dim=384, num_heads=8)

    def make_model(target_type: str):
        video = WanVideoDiT(**config)
        action = WanVideoDiT(**config)
        mot = MoT(
            mixtures={"video": video, "action": action},
            mot_checkpoint_mixed_attn=False,
        )
        encoder = None
        teacher_mode = "frozen_pretrained"
        if target_type == "patch_embedding":
            teacher_mode = "frozen_random"
            encoder = build_random_foresight_encoder(
                config,
                max_layers=0,
                seed=42,
                device="cpu",
                torch_dtype=torch.float32,
            )
        model = JAMWan(
            video_expert=video,
            action_expert=action,
            mot=mot,
            vae=TinyVAE(),
            text_dim=32,
            device="cpu",
            torch_dtype=torch.float32,
            foresight_encoder=encoder,
            foresight_cfg={
                "enabled": True,
                "video_cotrain": False,
                "target_type": target_type,
                "teacher_mode": teacher_mode,
                "layers": [2],
                "encoder_layers": [0],
                "offsets": [0, 1, 2],
                "per_frame": True,
                "shared_projector": True,
                "projector_hidden_dim": 96,
                "projector_num_layers": 1,
                "projector_num_heads": 4,
                "projector_ffn_dim": 192,
                "action_layers": None,
            },
        )
        return model

    for target_type in ("patch_embedding", "vae_direct"):
        model = make_model(target_type)
        latents = torch.randn(2, 48, 3, 8, 8)
        student = torch.randn(2, 16, 192, requires_grad=True)
        loss, action_loss = model._compute_foresight_loss(
            video_hidden_by_layer={2: student},
            action_hidden_by_layer=None,
            tokens_per_frame=16,
            input_latents=latents,
            foresight_target_latents=None,
            context=torch.randn(2, 5, 32),
            context_mask=torch.ones(2, 5, dtype=torch.bool),
            fuse_vae_embedding_in_latents=True,
            image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
        )
        assert torch.isfinite(loss) and action_loss.item() == 0.0
        loss.backward()
        assert student.grad is not None and student.grad.abs().sum() > 0
        assert all(
            parameter.grad is None
            for parameter in (model.foresight_encoder.parameters() if model.foresight_encoder else [])
        )
        print(f"[ok] JAMWan target dispatch + gradient: {target_type}, loss={loss.item():.4f}")


def test_cached_vae_direct_loss_equivalence():
    """Compact raw VAE cache must reproduce the online VAE-direct loss."""
    config = tiny_video_dit_config(num_layers=4)
    config.update(hidden_dim=192, ffn_dim=384, num_heads=8)
    video = WanVideoDiT(**config)
    action = WanVideoDiT(**config)
    model = JAMWan(
        video_expert=video,
        action_expert=action,
        mot=MoT(
            mixtures={"video": video, "action": action},
            mot_checkpoint_mixed_attn=False,
        ),
        vae=type("TinyVAE", (torch.nn.Module,), {"temporal_downsample_factor": 4})(),
        text_dim=32,
        device="cpu",
        torch_dtype=torch.float32,
        foresight_cfg={
            "enabled": True,
            "video_cotrain": False,
            "target_type": "vae_direct",
            "teacher_mode": "frozen_pretrained",
            "target_cache_roots": ["synthetic-compact-cache"],
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
        },
    )
    latents = torch.randn(2, 48, 3, 8, 8)
    raw, tpf = patch_foresight_vae_latents(latents, patch_size=(1, 2, 2))
    student = torch.randn(2, tpf, 192)
    kwargs = dict(
        video_hidden_by_layer={2: student},
        action_hidden_by_layer=None,
        tokens_per_frame=tpf,
        input_latents=latents,
        foresight_target_latents=None,
        context=torch.randn(2, 5, 32),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True,
        image_is_pad=torch.zeros(2, 9, dtype=torch.bool),
    )
    online, _ = model._compute_foresight_loss(**kwargs)
    cached, _ = model._compute_foresight_loss(
        **kwargs,
        precomputed_enc_cache={0: raw},
    )
    assert torch.equal(online, cached), (online, cached)
    print(f"[ok] compact cached VAE-direct == online loss exactly: {online.item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_projector()
    test_encoder_and_fusion()
    test_teacher_variants_and_ema()
    test_vae_direct_target()
    test_random_teacher_vae_builder()
    test_mot_hidden_capture()
    test_gradient_flow()
    test_jam_wan_target_dispatch()
    test_cached_vae_direct_loss_equivalence()
    print("\nAll foresight smoke tests passed.")
