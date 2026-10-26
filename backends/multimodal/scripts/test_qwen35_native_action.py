#!/usr/bin/env python3
"""CPU structural/numerical checks for the Qwen-native Action Expert."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from jam_multimodal.models.backbones.action_dit_qwen35 import (
    ActionDiTQwen35Native,
    QWEN_ACTION_ATTENTION_CAUSAL,
)


def build_tiny_source(attention_implementation: str = "eager"):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormGated,
        Qwen3_5TextRotaryEmbedding,
        torch_chunk_gated_delta_rule,
    )

    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [4, 2, 2],
        },
    )
    config._attn_implementation = str(attention_implementation)

    class TinyLanguage(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [Qwen3_5DecoderLayer(config, idx) for idx in range(4)]
            )
            # This is deliberately a CPU numerical test.  When optional CUDA
            # kernels are installed, Transformers binds CUDA-only FLA and
            # causal-conv callables even though these tiny modules remain on
            # CPU.  Pin the tiny reference and its copied Action blocks to the
            # official Torch fallbacks so the result is independent of which
            # optional CUDA extensions happen to be importable.
            for layer in self.layers:
                if layer.block_type == "linear_attention":
                    layer.linear_attn.causal_conv1d_fn = None
                    layer.linear_attn.chunk_gated_delta_rule = (
                        torch_chunk_gated_delta_rule
                    )
                    torch_norm = Qwen3_5RMSNormGated(
                        layer.linear_attn.head_v_dim,
                        eps=layer.linear_attn.layer_norm_epsilon,
                    )
                    torch_norm.load_state_dict(
                        layer.linear_attn.norm.state_dict()
                    )
                    layer.linear_attn.norm = torch_norm
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = TinyLanguage()
            self.config = SimpleNamespace(text_config=config)

    return TinyBackbone(), config


def assert_source_copy_is_exact(source, action):
    source_state = source.language_model.layers.state_dict()
    action_state = action.blocks.state_dict()
    assert source_state.keys() == action_state.keys()
    for key in source_state:
        assert torch.equal(source_state[key], action_state[key]), key
        assert source_state[key].data_ptr() != action_state[key].data_ptr(), key
    for key, value in source.language_model.norm.state_dict().items():
        copied = action.final_norm.state_dict()[key]
        assert torch.equal(value, copied), key
        assert value.data_ptr() != copied.data_ptr(), key


def build_condition(action, prefixes):
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    positions = torch.arange(6, dtype=torch.long)[None, None].expand(3, 2, -1).clone()
    proprio = torch.randn(2, 1, action.hidden_dim, requires_grad=True)
    return action.prepare_condition(
        layerwise_prefix_tokens=prefixes,
        prefix_attention_mask=mask,
        prefix_position_ids=positions,
        proprio_token=proprio,
    )


def test_forward_backward_and_copy():
    torch.manual_seed(7)
    source, config = build_tiny_source()
    action = ActionDiTQwen35Native.from_qwen_backbone(
        source,
        action_dim=7,
        max_action_horizon=16,
        timestep_embed_dim=32,
    )
    assert tuple(action.layer_types) == tuple(config.layer_types)
    assert_source_copy_is_exact(source, action)

    prefixes = tuple(
        torch.randn(2, 6, config.hidden_size, requires_grad=True)
        for _ in range(4)
    )
    condition = build_condition(action, prefixes)
    noisy_action = torch.randn(2, 5, 7, requires_grad=True)
    output = action(noisy_action, torch.tensor([10.0, 700.0]), condition)
    assert output.shape == noisy_action.shape
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert noisy_action.grad is not None and noisy_action.grad.abs().sum() > 0
    assert condition.proprio_token.grad is not None
    for layer_idx, prefix in enumerate(prefixes):
        assert prefix.grad is not None, layer_idx
        assert torch.isfinite(prefix.grad).all(), layer_idx
        assert prefix.grad.abs().sum() > 0, layer_idx
    missing = [
        name
        for name, parameter in action.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing


def _test_discarded_prefix_optimization_equivalence(
    attention_implementation: str,
):
    """All dead-prefix eliminations must match the full-row graph."""

    torch.manual_seed(29)
    source, config = build_tiny_source(attention_implementation)
    reference = ActionDiTQwen35Native.from_qwen_backbone(
        source,
        action_dim=7,
        max_action_horizon=16,
        timestep_embed_dim=32,
        skip_discarded_prefix_mlp=False,
    )
    optimized = copy.deepcopy(reference)
    optimized.skip_discarded_prefix_mlp = True
    query_optimized = copy.deepcopy(optimized)
    query_optimized.skip_discarded_prefix_query = True

    base_prefixes = tuple(torch.randn(2, 6, config.hidden_size) for _ in range(4))
    base_action = torch.randn(2, 5, 7)
    base_proprio = torch.randn(2, 1, config.hidden_size)
    timestep = torch.tensor([75.0, 425.0])

    def run(model):
        prefixes = tuple(item.detach().clone().requires_grad_(True) for item in base_prefixes)
        noisy_action = base_action.detach().clone().requires_grad_(True)
        condition = model.prepare_condition(
            layerwise_prefix_tokens=prefixes,
            prefix_attention_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
            ),
            prefix_position_ids=torch.arange(6, dtype=torch.long)[None, None]
            .expand(3, 2, -1)
            .clone(),
            proprio_token=base_proprio.detach().clone().requires_grad_(True),
        )
        output = model(noisy_action, timestep, condition)
        output.float().square().mean().backward()
        parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return output.detach(), noisy_action.grad.detach(), prefixes, condition, parameter_grads

    ref_output, ref_action_grad, ref_prefixes, ref_condition, ref_parameter_grads = run(reference)
    opt_output, opt_action_grad, opt_prefixes, opt_condition, opt_parameter_grads = run(optimized)
    (
        query_output,
        query_action_grad,
        query_prefixes,
        query_condition,
        query_parameter_grads,
    ) = run(query_optimized)

    torch.testing.assert_close(ref_output, opt_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ref_action_grad, opt_action_grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        ref_condition.proprio_token.grad,
        opt_condition.proprio_token.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    for ref_prefix, opt_prefix in zip(ref_prefixes, opt_prefixes):
        torch.testing.assert_close(ref_prefix.grad, opt_prefix.grad, atol=2e-5, rtol=2e-5)
    assert ref_parameter_grads.keys() == opt_parameter_grads.keys()
    for name in ref_parameter_grads:
        ref_grad = ref_parameter_grads[name]
        opt_grad = opt_parameter_grads[name]
        assert (ref_grad is None) == (opt_grad is None), name
        if ref_grad is not None:
            torch.testing.assert_close(ref_grad, opt_grad, atol=2e-5, rtol=2e-5)

    # The GatedDeltaNet Q-pruned path is compared with the already equivalent
    # suffix-only graph.  This includes both variable prefix lengths in the
    # batch, the proprio row, the causal-convolution left context, every input
    # gradient, and every trainable parameter gradient.
    torch.testing.assert_close(opt_output, query_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        opt_action_grad, query_action_grad, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        opt_condition.proprio_token.grad,
        query_condition.proprio_token.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    for opt_prefix, query_prefix in zip(opt_prefixes, query_prefixes):
        torch.testing.assert_close(
            opt_prefix.grad, query_prefix.grad, atol=2e-5, rtol=2e-5
        )
    assert opt_parameter_grads.keys() == query_parameter_grads.keys()
    for name in opt_parameter_grads:
        opt_grad = opt_parameter_grads[name]
        query_grad = query_parameter_grads[name]
        assert (opt_grad is None) == (query_grad is None), name
        if opt_grad is not None:
            torch.testing.assert_close(
                opt_grad, query_grad, atol=2e-5, rtol=2e-5
            )


def test_discarded_prefix_mlp_optimization_equivalence():
    for attention_implementation in ("eager", "sdpa"):
        _test_discarded_prefix_optimization_equivalence(
            attention_implementation
        )


@torch.no_grad()
def test_directionality_and_padding():
    torch.manual_seed(11)
    source, config = build_tiny_source()
    action = ActionDiTQwen35Native.from_qwen_backbone(
        source,
        action_dim=7,
        max_action_horizon=16,
        timestep_embed_dim=32,
    ).eval()
    assert action.action_attention_mode == QWEN_ACTION_ATTENTION_CAUSAL
    prefixes = tuple(torch.randn(2, 6, config.hidden_size) for _ in range(4))
    condition = build_condition(action, prefixes)
    noisy_action = torch.randn(2, 5, 7)
    timestep = torch.tensor([50.0, 500.0])
    reference = action(noisy_action, timestep, condition)

    changed_late = noisy_action.clone()
    changed_late[:, -1] += 9.0
    late_output = action(changed_late, timestep, condition)
    torch.testing.assert_close(reference[:, :-1], late_output[:, :-1], atol=2e-5, rtol=2e-5)
    assert not torch.allclose(reference[:, -1], late_output[:, -1])

    changed_prefixes = list(copy.deepcopy(prefixes))
    changed_prefixes[0][:, 0] += 3.0
    prefix_output = action(
        noisy_action,
        timestep,
        action.prepare_condition(
            layerwise_prefix_tokens=changed_prefixes,
            prefix_attention_mask=condition.prefix_attention_mask,
            prefix_position_ids=condition.prefix_position_ids,
            proprio_token=condition.proprio_token,
        ),
    )
    assert not torch.allclose(reference, prefix_output)

    changed_padding = list(copy.deepcopy(prefixes))
    for prefix in changed_padding:
        prefix[1, 4:] = 1_000.0
    padding_output = action(
        noisy_action,
        timestep,
        action.prepare_condition(
            layerwise_prefix_tokens=changed_padding,
            prefix_attention_mask=condition.prefix_attention_mask,
            prefix_position_ids=condition.prefix_position_ids,
            proprio_token=condition.proprio_token,
        ),
    )
    torch.testing.assert_close(reference, padding_output, atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_unknown_action_attention_mode_fails_loudly():
    source, _ = build_tiny_source()
    try:
        ActionDiTQwen35Native.from_qwen_backbone(
            source,
            action_dim=7,
            max_action_horizon=16,
            timestep_embed_dim=32,
            action_attention_mode="fully_bidirectional_everywhere",
        )
    except ValueError as exc:
        assert "action_attention_mode" in str(exc)
    else:
        raise AssertionError("Unknown Qwen action attention modes must fail loudly.")


def main():
    test_forward_backward_and_copy()
    test_discarded_prefix_mlp_optimization_equivalence()
    test_directionality_and_padding()
    test_unknown_action_attention_mode_fails_loudly()
    print("QWEN35 NATIVE ACTION CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
