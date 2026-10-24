#!/usr/bin/env python3
"""CPU regression checks for the Ovis frozen-random teacher builder."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from jam_multimodal.models.backbones.foresight_yak import (
    YakForesightTeacher,
    _build_random_yak_expert,
    _normalize_teacher_mode,
)
from jam_multimodal.models.backbones.ovis_u1_imports import ensure_ovis_u1_remote_code_importable
from jam_multimodal.models.backbones.ovis_u1_video_expert import OvisU1VideoExpert

ensure_ovis_u1_remote_code_importable()

from ovis_u1_hf.configuration_yak import YakConfig  # noqa: E402
from ovis_u1_hf.modeling_yak import YakTransformer  # noqa: E402


class ToyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)


class ToyExpert(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten() for parameter in module.parameters()])


def main() -> None:
    torch.manual_seed(7)
    source = ToyExpert(ToyTransformer(SimpleNamespace(hidden_size=8)))
    source_before = flat_params(source).clone()

    state_before = torch.random.get_rng_state()
    random_a = _build_random_yak_expert(source, seed=42)
    state_after = torch.random.get_rng_state()
    random_b = _build_random_yak_expert(source, seed=42)
    random_c = _build_random_yak_expert(source, seed=43)

    assert torch.equal(state_before, state_after), "random teacher shifted the caller RNG"
    assert torch.equal(source_before, flat_params(source)), "source expert was modified"
    assert torch.equal(flat_params(random_a), flat_params(random_b)), "same seed is not deterministic"
    assert not torch.equal(flat_params(random_a), flat_params(random_c)), "different seeds matched"
    assert not torch.equal(source_before, flat_params(random_a)), "random teacher copied pretrained weights"

    teacher = YakForesightTeacher(random_a)
    teacher.train(True)
    assert not teacher.training and not teacher.expert.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert _normalize_teacher_mode(None) == "frozen_pretrained"
    assert _normalize_teacher_mode(" FROZEN_RANDOM ") == "frozen_random"
    try:
        _normalize_teacher_mode("online_ema")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid teacher mode was accepted")

    # Exercise the real remote-code Yak constructor with a tiny topology. This
    # catches assumptions that a generic nn.Module test double cannot cover.
    yak_config = YakConfig(
        in_channels=16,
        out_channels=16,
        vec_in_dim=64,
        context_in_dim=64,
        hidden_size=64,
        mlp_ratio=2,
        num_heads=1,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[8, 28, 28],
        qkv_bias=True,
    )
    yak_source = OvisU1VideoExpert(YakTransformer(yak_config))
    yak_a = _build_random_yak_expert(yak_source, seed=42)
    yak_b = _build_random_yak_expert(yak_source, seed=42)
    assert isinstance(yak_a, OvisU1VideoExpert)
    assert torch.equal(flat_params(yak_a), flat_params(yak_b))
    assert not torch.equal(flat_params(yak_source), flat_params(yak_a))

    print("OVIS_RANDOM_TEACHER_OK")


if __name__ == "__main__":
    main()
