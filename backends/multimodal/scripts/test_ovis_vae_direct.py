#!/usr/bin/env python3
"""CPU contract tests for the Ovis pretrained-VAE-direct target."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jam_multimodal.models.backbones.foresight_yak import (
    ForesightProjector,
    attach_ovis_foresight,
    lift_vae_direct_tokens,
)
from jam_multimodal.models.backbones.jam import JAMPolicy


class _DummyVideoExpert(nn.Module):
    hidden_dim = 1536
    double_layers = 6
    single_layers = 12

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_expert = _DummyVideoExpert()
        self.mot = nn.Module()


def test_lift_isometry() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 11, 16)
    y = torch.randn(3, 11, 16)
    lx = lift_vae_direct_tokens(x, output_dim=1536)
    ly = lift_vae_direct_tokens(y, output_dim=1536)
    assert lx.shape == (3, 11, 1536)
    torch.testing.assert_close(lx.norm(dim=-1), x.norm(dim=-1), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        (lx * ly).sum(dim=-1),
        (x * y).sum(dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        F.cosine_similarity(lx, ly, dim=-1),
        F.cosine_similarity(x, y, dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_attach_contract() -> None:
    model = _DummyModel()
    attach_ovis_foresight(
        model,
        {
            "enabled": True,
            "teacher_type": "vae_direct",
            "teacher_mode": "frozen_pretrained",
            "drop_target_tokens": True,
            "layer": 11,
            "encoder_layer": 11,
            "teacher_early_exit": False,
            "offsets": [1],
            "lambda_foresight": 0.3,
            "projector_hidden_dim": 768,
            "projector_ffn_dim": 3072,
            "projector_num_layers": 2,
            "projector_num_heads": 12,
        },
    )
    assert model.foresight_teacher_type == "vae_direct"
    assert model.foresight_offsets == [1]
    assert model.foresight_layer == 11
    assert math.isclose(model.foresight_lambda, 0.3)
    assert model.foresight_vae_direct_output_dim == 1536
    assert "foresight_teacher" not in model.__dict__
    assert model.mot.foresight_projector.heads[0].out_features == 1536


def test_loss_and_gradient() -> None:
    model = JAMPolicy.__new__(JAMPolicy)
    nn.Module.__init__(model)
    model.foresight_layer = 11
    model.foresight_teacher_type = "vae_direct"
    model.foresight_vae_direct_output_dim = 1536
    model.mot = nn.Module()
    model.mot.foresight_projector = ForesightProjector(
        in_dim=1536,
        hidden_dim=64,
        ffn_dim=128,
        num_layers=1,
        num_heads=1,
        num_targets=1,
        out_dim=1536,
    )
    student = torch.randn(2, 9, 1536, requires_grad=True)
    target = torch.randn(2, 9, 16)
    loss = model._compute_ovis_foresight_loss(
        inputs={"target_latent": target},
        video_pre={"target_len": 0},
        captured={11: student},
        batch_size=2,
    )
    assert loss.ndim == 0 and bool(torch.isfinite(loss).item())
    loss.backward()
    assert student.grad is not None and bool(torch.isfinite(student.grad).all().item())
    assert target.grad is None
    grads = [p.grad for p in model.mot.foresight_projector.parameters()]
    assert grads and all(g is not None and bool(torch.isfinite(g).all().item()) for g in grads)


def main() -> None:
    test_lift_isometry()
    test_attach_contract()
    test_loss_and_gradient()
    print("OVIS VAE-DIRECT CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
