"""CPU checks for the shared-weight Qwen action stream (no separate action expert).

Run from backends/multimodal in a Transformers-5 environment:
    PYTHONPATH=src python scripts/test_qwen35_shared_action.py

A tiny random Qwen3.5 text model stands in for the backbone.  The checks:
  1. share_backbone_weights=True reuses the backbone's block/norm/rotary modules;
  2. at initialization the shared stream and the MoT copy give identical actions;
  3. the gradient of every shared weight equals backbone-grad + expert-copy-grad
     of the MoT pair, i.e. weight tying is exact;
  4. deduplicated parameter counts drop by exactly the block/norm copy size;
  5. a state_dict saved from a shared model loads strictly into a fresh shared
     model and reproduces its actions;
  6. AdamW over ``parameters()`` sees each shared tensor once and keeps it tied.
"""

from __future__ import annotations

import copy
import io
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_modeling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from jam_multimodal.models.backbones.action_dit_qwen35 import ActionDiTQwen35Native


def tiny_text_config() -> Qwen3_5TextConfig:
    ref_path = os.path.join("checkpoints", "Qwen3.5-2B-Base", "config.json")
    with open(ref_path, encoding="utf-8") as handle:
        text = dict(json.load(handle)["text_config"])
    text.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=128,
    )
    text.pop("model_type", None)
    config = Qwen3_5TextConfig(**text)
    rope = dict(getattr(config, "rope_parameters", None) or {})
    rotary_dim = int(config.head_dim * float(rope.get("partial_rotary_factor", 1.0)))
    if "mrope_section" in rope:
        half = rotary_dim // 2
        rope["mrope_section"] = [half - 2 * (half // 3), half // 3, half // 3]
        config.rope_parameters = rope
    return config


class TinyBackbone(nn.Module):
    def __init__(self, text_config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.language_model = qwen_modeling.Qwen3_5TextModel(text_config)
        self.config = SimpleNamespace(text_config=text_config)


def use_torch_kernels(module: nn.Module) -> None:
    """CUDA-only FLA/causal-conv kernels -> reference torch paths on CPU."""

    for sub in module.modules():
        if hasattr(sub, "chunk_gated_delta_rule"):
            sub.chunk_gated_delta_rule = qwen_modeling.torch_chunk_gated_delta_rule
        if hasattr(sub, "causal_conv1d_fn"):
            sub.causal_conv1d_fn = None


def layer_outputs(backbone: TinyBackbone, embeds, mask, position_ids):
    """Per-layer outputs with the last replaced by the final-norm output, as the
    policy wrapper passes them to the action stream."""

    captured = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out))
        for layer in backbone.language_model.layers
    ]
    try:
        output = backbone.language_model(
            inputs_embeds=embeds,
            attention_mask=mask.to(torch.long),
            position_ids=position_ids,
            use_cache=False,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == len(backbone.language_model.layers)
    return (*captured[:-1], output.last_hidden_state)


def action_forward(backbone, expert, embeds, mask, position_ids, actions, timestep):
    prefixes = layer_outputs(backbone, embeds, mask, position_ids)
    condition = expert.prepare_condition(
        layerwise_prefix_tokens=prefixes,
        prefix_attention_mask=mask,
        prefix_position_ids=position_ids,
    )
    return expert(actions, timestep, condition)


def unique_numel(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    torch.manual_seed(0)
    text_config = tiny_text_config()
    kwargs = dict(action_dim=7, max_action_horizon=8, timestep_embed_dim=32, torch_dtype=torch.float32)

    backbone_shared = TinyBackbone(text_config).float()
    use_torch_kernels(backbone_shared)
    backbone_mot = copy.deepcopy(backbone_shared)
    expert_shared = ActionDiTQwen35Native.from_qwen_backbone(backbone_shared, share_backbone_weights=True, **kwargs)
    expert_mot = ActionDiTQwen35Native.from_qwen_backbone(backbone_mot, share_backbone_weights=False, **kwargs)
    use_torch_kernels(expert_mot)
    # identical action-specific heads so that the two policies start equal
    for name in ("action_encoder", "time_in", "head"):
        getattr(expert_mot, name).load_state_dict(getattr(expert_shared, name).state_dict())
    expert_mot.action_position.data.copy_(expert_shared.action_position.data)

    # 1. module identity
    layers = backbone_shared.language_model.layers
    assert expert_shared.shares_backbone_weights and not expert_mot.shares_backbone_weights
    assert all(a is b for a, b in zip(expert_shared.blocks, layers))
    assert expert_shared.final_norm is backbone_shared.language_model.norm
    assert expert_shared.rotary_emb is backbone_shared.language_model.rotary_emb
    assert all(a is not b for a, b in zip(expert_mot.blocks, backbone_mot.language_model.layers))
    print("[1] shared stream reuses the backbone's blocks, final norm and rotary embedding")

    # 2. identical outputs at initialization (right-padded batch)
    batch, prefix_len, horizon = 2, 6, 4
    embeds = torch.randn(batch, prefix_len, text_config.hidden_size)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    positions = torch.arange(prefix_len)[None, None].expand(3, batch, prefix_len).contiguous()
    actions = torch.randn(batch, horizon, 7)
    timestep = torch.tensor([0.3, 0.8])
    out_shared = action_forward(backbone_shared, expert_shared, embeds, mask, positions, actions, timestep)
    out_mot = action_forward(backbone_mot, expert_mot, embeds, mask, positions, actions, timestep)
    max_diff = (out_shared - out_mot).abs().max().item()
    assert torch.allclose(out_shared, out_mot, atol=1e-6, rtol=0), max_diff
    print(f"[2] identical actions at initialization (max |diff| = {max_diff:.2e})")

    # 3. exact weight tying in the gradients
    target = torch.randn_like(out_shared)
    ((out_shared - target) ** 2).mean().backward()
    ((out_mot - target) ** 2).mean().backward()
    worst = 0.0
    checked = 0
    for idx, (layer_shared, layer_mot, copy_mot) in enumerate(
        zip(layers, backbone_mot.language_model.layers, expert_mot.blocks)
    ):
        grads_backbone = dict(layer_mot.named_parameters())
        grads_copy = dict(copy_mot.named_parameters())
        for name, parameter in layer_shared.named_parameters():
            reference = torch.zeros_like(parameter)
            for source in (grads_backbone[name], grads_copy[name]):
                if source.grad is not None:
                    reference = reference + source.grad
            got = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
            worst = max(worst, (got - reference).abs().max().item())
            assert torch.allclose(got, reference, atol=1e-6, rtol=1e-4), (idx, name)
            checked += 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layers[-1].parameters())
    print(f"[3] shared grad == backbone grad + expert-copy grad for {checked} tensors (max |diff| = {worst:.2e})")

    # 4. deduplicated parameter counts
    dit_shared = nn.ModuleDict({"qwen": backbone_shared, "action_expert": expert_shared})
    dit_mot = nn.ModuleDict({"qwen": backbone_mot, "action_expert": expert_mot})
    copy_size = unique_numel(backbone_mot.language_model.layers) + unique_numel(backbone_mot.language_model.norm)
    assert unique_numel(dit_mot) - unique_numel(dit_shared) == copy_size
    print(f"[4] unique params: MoT={unique_numel(dit_mot)} shared={unique_numel(dit_shared)} (difference = block+norm copy {copy_size})")

    # 5. strict state_dict round trip into a fresh shared model
    buffer = io.BytesIO()
    torch.save(dit_shared.state_dict(), buffer)
    torch.manual_seed(123)
    fresh_backbone = TinyBackbone(text_config).float()
    use_torch_kernels(fresh_backbone)
    fresh_expert = ActionDiTQwen35Native.from_qwen_backbone(fresh_backbone, share_backbone_weights=True, **kwargs)
    fresh = nn.ModuleDict({"qwen": fresh_backbone, "action_expert": fresh_expert})
    buffer.seek(0)
    fresh.load_state_dict(torch.load(buffer), strict=True)
    assert all(a is b for a, b in zip(fresh_expert.blocks, fresh_backbone.language_model.layers))
    with torch.no_grad():
        again = action_forward(fresh_backbone, fresh_expert, embeds, mask, positions, actions, timestep)
    assert torch.allclose(again, out_shared.detach(), atol=1e-6, rtol=0)
    print("[5] state_dict round trip is strict and reproduces the actions; blocks stay tied")

    # 6. optimizer sees every tensor once and the tie survives a step
    optimizer = torch.optim.AdamW(dit_shared.parameters(), lr=1e-3)
    n_opt = sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"])
    assert n_opt == unique_numel(dit_shared)
    watched = layers[0].mlp.down_proj.weight
    before = watched.detach().clone()
    optimizer.step()
    assert expert_shared.blocks[0].mlp.down_proj.weight is watched
    assert not torch.equal(before, watched.detach())
    print(f"[6] AdamW holds {n_opt} unique parameters once; shared weights stay tied after a step")

    single_stream_check(text_config, kwargs)
    print("QWEN35 SHARED ACTION CPU TEST PASSED")


def layer_inputs(backbone: TinyBackbone, embeds, mask, position_ids):
    captured = {}
    hooks = [
        layer.register_forward_pre_hook(lambda _m, args, _i=i: captured.__setitem__(_i, args[0]))
        for i, layer in enumerate(backbone.language_model.layers)
    ]
    try:
        backbone.language_model(
            inputs_embeds=embeds, attention_mask=mask.to(torch.long),
            position_ids=position_ids, use_cache=False,
        )
    finally:
        for hook in hooks:
            hook.remove()
    return tuple(captured[i] for i in range(len(backbone.language_model.layers)))


def single_stream_check(text_config, kwargs) -> None:
    """7. single_stream == one causal Qwen pass over [prefix; proprio; actions]."""

    torch.manual_seed(7)
    backbone = TinyBackbone(text_config).float()
    use_torch_kernels(backbone)
    expert = ActionDiTQwen35Native.from_qwen_backbone(
        backbone, share_backbone_weights=True, single_stream=True, **kwargs
    )
    batch, prefix_len, horizon, dim = 2, 6, 4, text_config.hidden_size
    embeds = torch.randn(batch, prefix_len, dim)
    mask = torch.ones(batch, prefix_len, dtype=torch.bool)
    positions = torch.arange(prefix_len)[None, None].expand(3, batch, prefix_len).contiguous()
    proprio = torch.randn(batch, 1, dim)
    actions = torch.randn(batch, horizon, 7)
    timestep = torch.tensor([0.25, 0.9])
    with torch.no_grad():
        condition = expert.prepare_condition(
            layerwise_prefix_tokens=layer_inputs(backbone, embeds, mask, positions),
            prefix_attention_mask=mask, prefix_position_ids=positions, proprio_token=proprio,
        )
        out_stream = expert(actions, timestep, condition)
        sequence = torch.cat((embeds, proprio, expert._embed_action(actions, timestep)), dim=1)
        total = sequence.shape[1]
        joint_positions = torch.arange(total)[None, None].expand(3, batch, total).contiguous()
        joint = backbone.language_model(
            inputs_embeds=sequence, attention_mask=torch.ones(batch, total, dtype=torch.long),
            position_ids=joint_positions, use_cache=False,
        ).last_hidden_state
        out_joint = expert.head(joint[:, prefix_len + 1:])
        diff = (out_stream - out_joint).abs().max().item()
        assert torch.allclose(out_stream, out_joint, atol=1e-5, rtol=1e-4), diff
        # the MoT-aligned shared stream (layer outputs, per-layer proprio) is a different model
        expert.single_stream = False
        outputs = layer_outputs(backbone, embeds, mask, positions)
        tied_condition = expert.prepare_condition(
            layerwise_prefix_tokens=outputs, prefix_attention_mask=mask,
            prefix_position_ids=positions, proprio_token=proprio,
        )
        gap = (expert(actions, timestep, tied_condition) - out_joint).abs().max().item()
        expert.single_stream = True
    print(f"[7] single_stream equals one causal Qwen pass over [prefix; proprio; actions] "
          f"(max |diff| = {diff:.2e}); the MoT-aligned shared stream differs by {gap:.2e}")


if __name__ == "__main__":
    main()
