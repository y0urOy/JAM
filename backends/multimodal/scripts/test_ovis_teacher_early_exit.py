#!/usr/bin/env python3
"""CPU regression checks for the frozen Yak teacher's exact early exit."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn

from jam_multimodal.models.backbones.mot import MoT


class _Modulation:
    def __init__(self, reference: torch.Tensor, gate: float = 1.0):
        self.scale = torch.zeros_like(reference)
        self.shift = torch.zeros_like(reference)
        self.gate = torch.full_like(reference, gate)


class _Scale(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.value


class _ToyAttention(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.proj = _Scale(value)
        self.qkv_scale = float(value)


class _ToyDoubleBlock(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        scale = 0.05 * float(layer)
        self.executions = 0
        self.img_norm1 = nn.Identity()
        self.img_norm2 = nn.Identity()
        self.txt_norm1 = nn.Identity()
        self.txt_norm2 = nn.Identity()
        self.img_attn = _ToyAttention(scale)
        self.img_self_attn = _ToyAttention(scale * 0.5)
        self.txt_attn = _ToyAttention(scale)
        self.img_mlp = _Scale(scale)
        self.txt_mlp = _Scale(scale)

    def img_mod(self, vec: torch.Tensor):
        self.executions += 1
        return _Modulation(vec), _Modulation(vec), _Modulation(vec)

    @staticmethod
    def txt_mod(vec: torch.Tensor):
        return _Modulation(vec), _Modulation(vec)


class _ToySingleBlock(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.executions = 0
        self.scale = 0.05 * float(layer)


class _ToyMoT:
    """Reuse the production control flow with deterministic tiny block math."""

    block_protocol = "yak"
    forward_yak_video_only = MoT.forward_yak_video_only

    @staticmethod
    def _yak_qkv(attn, tensor: torch.Tensor, _pe: torch.Tensor):
        value = tensor * attn.qkv_scale
        return value, value, value

    @staticmethod
    def _mixed_attention(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        _mask: torch.Tensor,
    ) -> torch.Tensor:
        return q

    @staticmethod
    def _yak_single_io(block, tensor: torch.Tensor, _vec: torch.Tensor, _pe: torch.Tensor):
        block.executions += 1
        value = tensor * block.scale
        return {"q": value, "k": value, "v": value, "residual_x": tensor}

    @staticmethod
    def _yak_single_post(_block, mixed: torch.Tensor, state: dict) -> torch.Tensor:
        return state["residual_x"] + mixed


def _make_expert(double_layers: int = 2, single_layers: int = 3):
    return SimpleNamespace(
        double_layers=double_layers,
        single_layers=single_layers,
        double_blocks=nn.ModuleList(
            [_ToyDoubleBlock(layer + 1) for layer in range(double_layers)]
        ),
        single_blocks=nn.ModuleList(
            [
                _ToySingleBlock(double_layers + layer + 1)
                for layer in range(single_layers)
            ]
        ),
    )


def _inputs():
    torch.manual_seed(7)
    batch_size, txt_len, img_len, dim = 2, 2, 3, 4
    return {
        "video_tokens": {
            "txt": torch.randn(batch_size, txt_len, dim),
            "img": torch.randn(batch_size, img_len, dim),
        },
        "video_freqs": {
            "txt": torch.zeros(batch_size, 1, txt_len, 1),
            "img": torch.zeros(batch_size, 1, img_len, 1),
        },
        "video_t_mod": torch.randn(batch_size, 1, dim),
        "attention_mask": {
            "joint_double": torch.ones(1, dtype=torch.bool),
            "image_double": torch.ones(1, dtype=torch.bool),
            "single": torch.ones(1, dtype=torch.bool),
        },
    }


def _execution_counts(expert) -> list[int]:
    return [
        block.executions
        for block in [*expert.double_blocks, *expert.single_blocks]
    ]


def test_single_stream_early_exit_matches_full_capture():
    mot = _ToyMoT()
    # Mirror the real Ovis recipe exactly: 6 double + 12 single blocks, capture 11.
    full_expert = _make_expert(double_layers=6, single_layers=12)
    early_expert = deepcopy(full_expert)
    inputs = _inputs()

    full = mot.forward_yak_video_only(
        **inputs,
        capture_layers={11},
        video_expert=full_expert,
    )
    early = mot.forward_yak_video_only(
        **inputs,
        capture_layers={11},
        video_expert=early_expert,
        early_exit_after_last_capture=True,
    )

    assert full["executed_layers"] == 18
    assert early["executed_layers"] == 11
    assert _execution_counts(full_expert) == [1] * 18
    assert _execution_counts(early_expert) == [1] * 11 + [0] * 7
    assert torch.equal(early["captured"][11], full["captured"][11])
    assert torch.equal(early["final_video"]["img"], early["captured"][11])


def test_multiple_captures_stop_after_deepest():
    mot = _ToyMoT()
    full_expert = _make_expert(double_layers=2, single_layers=3)
    early_expert = deepcopy(full_expert)
    inputs = _inputs()

    full = mot.forward_yak_video_only(
        **inputs,
        capture_layers={2, 4},
        video_expert=full_expert,
    )
    early = mot.forward_yak_video_only(
        **inputs,
        capture_layers={2, 4},
        video_expert=early_expert,
        early_exit_after_last_capture=True,
    )

    assert early["executed_layers"] == 4
    assert _execution_counts(early_expert) == [1, 1, 1, 1, 0]
    assert torch.equal(early["captured"][2], full["captured"][2])
    assert torch.equal(early["captured"][4], full["captured"][4])


def test_double_stream_exit_and_validation():
    mot = _ToyMoT()
    expert = _make_expert(double_layers=2, single_layers=3)
    early = mot.forward_yak_video_only(
        **_inputs(),
        capture_layers={1},
        video_expert=expert,
        early_exit_after_last_capture=True,
    )
    assert early["executed_layers"] == 1
    assert _execution_counts(expert) == [1, 0, 0, 0, 0]
    assert torch.equal(early["final_video"]["img"], early["captured"][1])

    try:
        mot.forward_yak_video_only(
            **_inputs(),
            capture_layers=set(),
            video_expert=_make_expert(double_layers=2, single_layers=3),
            early_exit_after_last_capture=True,
        )
    except ValueError as exc:
        assert "requires at least one" in str(exc)
    else:
        raise AssertionError("Early exit without a capture layer must fail.")

    try:
        mot.forward_yak_video_only(
            **_inputs(),
            capture_layers={6},
            video_expert=_make_expert(double_layers=2, single_layers=3),
        )
    except ValueError as exc:
        assert "[1, 5]" in str(exc)
    else:
        raise AssertionError("An out-of-range capture layer must fail.")


def main():
    test_single_stream_early_exit_matches_full_capture()
    test_multiple_captures_stop_after_deepest()
    test_double_stream_exit_and_validation()
    print("OVIS YAK TEACHER EARLY-EXIT CHECKS PASSED")


if __name__ == "__main__":
    main()
