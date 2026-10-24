"""Audit a shared-weight Qwen3.5 JAM checkpoint (no separate action expert).

Environment: QWEN_FINAL_CKPT, QWEN_RESOLVED_CONFIG, EXPECT_LAYERS, EXPECT_HIDDEN,
FORESIGHT_LAYER, FORESIGHT_LAMBDA, MAX_STEPS, ACTION_ONLY.  Run it on the
final checkpoint and resolved config of a shared-action training run.
"""

from __future__ import annotations

import os
import re

import torch
import yaml


def main() -> None:
    path = os.environ["QWEN_FINAL_CKPT"]
    layers = int(os.environ["EXPECT_LAYERS"])
    hidden = int(os.environ["EXPECT_HIDDEN"])
    foresight_layer = int(os.environ["FORESIGHT_LAYER"])
    lam = float(os.environ["FORESIGHT_LAMBDA"])
    steps = int(os.environ["MAX_STEPS"])
    action_only = os.environ.get("ACTION_ONLY", "0") == "1"
    single_stream = os.environ.get("EXPECT_SINGLE_STREAM", "0") == "1"
    expected_init = (
        "single_stream_qwen35_decoder_blocks_v1" if single_stream else "shared_qwen35_decoder_blocks_v1"
    )

    with open(os.environ["QWEN_RESOLVED_CONFIG"], encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]
    assert model["action_dit_config"]["share_backbone_weights"] is True
    assert int(model["context_dim"]) == hidden
    assert int(model["foresight"]["layer"]) == foresight_layer
    assert bool(model["foresight"]["enabled"]) is (not action_only)
    assert model["action_dit_config"]["action_attention_mode"] == "causal"
    assert bool(model["action_dit_config"].get("single_stream", False)) is single_stream

    payload = torch.load(path, map_location="cpu", mmap=True)
    assert payload["format"] == "qwen35_imagewam_native_qwen24_v1", payload["format"]
    assert int(payload["step"]) == steps, payload["step"]
    assert payload["action_fusion_type"] == "qwen_native_mot"
    assert payload["action_expert_family"] == "ActionDiTQwen35Native"
    assert payload["action_initialization"] == expected_init, payload["action_initialization"]
    assert bool(payload.get("action_single_stream", False)) is single_stream
    assert payload["action_shares_backbone_weights"] is True
    assert int(payload["qwen_num_layers"]) == layers
    assert int(payload["action_num_layers"]) == layers
    assert tuple(payload["action_visual_layer_map"]) == tuple(range(layers))
    assert tuple(payload["action_layer_types"]) == (
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ) * (layers // 4)
    assert int(payload["action_hidden_dim"]) == hidden
    assert payload["action_attention_mode"] == "causal"
    assert int(payload["foresight_layer"]) == foresight_layer
    assert tuple(payload["future_frame_offsets"]) == (4,)
    assert float(payload["loss_lambda_foresight"]) == lam
    assert payload["foresight_predictor_config"] == {
        "hidden_dim": 1024, "ffn_dim": 4096, "num_layers": 2, "num_heads": 16,
    }
    state = payload["dit"]
    assert not any(key.startswith("teacher") for key in state)

    # every action-stream block tensor must be the policy Qwen's own tensor
    pattern = re.compile(r"^action_expert\.blocks\.(\d+)\.(.+)$")
    tied = 0
    for key, value in state.items():
        match = pattern.match(key)
        if match is None:
            continue
        twin = f"qwen.language_model.layers.{match.group(1)}.{match.group(2)}"
        assert twin in state, f"missing backbone twin for {key}"
        assert torch.equal(value, state[twin]), f"untied weights: {key}"
        tied += 1
    assert tied > 0, "no action-stream block tensors found"
    for suffix in ("weight",):
        a, b = f"action_expert.final_norm.{suffix}", f"qwen.language_model.norm.{suffix}"
        assert torch.equal(state[a], state[b]), "untied final norm"

    unique = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            unique.setdefault((value.data_ptr(), tuple(value.shape)), (key, value))
    bad = [
        key for key, value in unique.values()
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item()
    ]
    assert not bad, f"non-finite tensors: {bad[:20]}"
    n_unique = sum(value.numel() for _, value in unique.values() if value.is_floating_point())
    print(
        "QWEN35 SHARED ACTION CHECKPOINT CHECK PASSED "
        f"path={path} step={steps} layers={layers} hidden={hidden} foresight_layer={foresight_layer} "
        f"lambda={lam} action_only={action_only} single_stream={single_stream} "
        f"tied_block_tensors={tied} unique_float_params={n_unique}"
    )


if __name__ == "__main__":
    main()
