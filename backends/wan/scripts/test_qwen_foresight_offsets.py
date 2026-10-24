#!/usr/bin/env python
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn

from jam_wan.models.qwen35.jam_qwen35 import (
    QWEN_ACTION_FUSION_LAYERWISE_MOT,
    JAMQwen35,
    extract_grouped_visual_tokens,
    forward_with_decoder_layer_capture,
    future_target_valid_mask,
    resolve_teacher_frame_layout,
    truncate_decoder_for_exact_early_exit,
)
from jam_wan.models.wan22.action_dit import ActionDiT


def test_offset_layout():
    frames, targets = resolve_teacher_frame_layout([4, 8])
    assert frames == (0, 4, 8)
    assert targets == (1, 2)

    frames, targets = resolve_teacher_frame_layout([0, 4, 8])
    assert frames == (0, 4, 8)
    assert targets == (0, 1, 2)


def test_future_pad_mask():
    image_is_pad = torch.tensor(
        [
            [False, False, False, False, False, False, False, False, True],
            [False, False, False, False, True, True, True, True, True],
        ]
    )
    valid = future_target_valid_mask(image_is_pad, [4, 8])
    expected = torch.tensor([[True, False], [False, False]])
    assert torch.equal(valid, expected)

    no_pad = future_target_valid_mask(None, [4, 8], batch_size=2, device=torch.device("cpu"))
    assert torch.equal(no_pad, torch.ones((2, 2), dtype=torch.bool))


class AddLayer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + self.value


class FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0), AddLayer(4.0)])


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeLanguageModel()

    def forward(self, input_ids, **_kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden + 10.0)


def test_one_based_layer_capture():
    backbone = FakeBackbone()
    model_inputs = {"input_ids": torch.tensor([[0, 1, 2]])}
    output, layer2 = forward_with_decoder_layer_capture(
        backbone,
        layer=2,
        model_inputs=model_inputs,
    )
    assert torch.equal(layer2.squeeze(-1), torch.tensor([[3.0, 4.0, 5.0]]))
    assert torch.equal(output.last_hidden_state.squeeze(-1), torch.tensor([[17.0, 18.0, 19.0]]))

    _, final_hidden = forward_with_decoder_layer_capture(
        backbone,
        layer=None,
        model_inputs=model_inputs,
    )
    assert torch.equal(final_hidden.squeeze(-1), torch.tensor([[17.0, 18.0, 19.0]]))

    _, _, decoder_inputs = forward_with_decoder_layer_capture(
        backbone,
        layer=2,
        model_inputs=model_inputs,
        return_decoder_inputs=True,
    )
    assert len(decoder_inputs) == 3
    assert torch.equal(decoder_inputs[0].squeeze(-1), torch.tensor([[0.0, 1.0, 2.0]]))
    assert torch.equal(decoder_inputs[1].squeeze(-1), torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(decoder_inputs[2].squeeze(-1), torch.tensor([[3.0, 4.0, 5.0]]))


def test_action_dit_layerwise_mot_mixes_and_masks_visual_tokens():
    torch.manual_seed(7)
    action_expert = ActionDiT(
        hidden_dim=8,
        action_dim=2,
        ffn_dim=16,
        text_dim=8,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=1,
        attn_head_dim=2,
        num_layers=3,
        use_gradient_checkpointing=True,
    )
    action = torch.randn(2, 4, 2)
    timestep = torch.rand(2)
    context = torch.randn(2, 5, 8)
    visual_contexts = tuple(
        torch.randn(2, 6, 8, requires_grad=True) for _ in range(3)
    )
    visual_tokens = action_expert.prepare_layerwise_mot_visual_tokens(visual_contexts)
    output = action_expert.forward_layerwise_mot(
        action_tokens=action,
        timestep=timestep,
        context=context,
        layerwise_visual_tokens=visual_tokens,
    )
    assert output.shape == action.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(item.grad is not None and torch.isfinite(item.grad).all() for item in visual_contexts)

    action_expert.eval()
    with torch.no_grad():
        zero_visual = action_expert.prepare_layerwise_mot_visual_tokens(
            tuple(torch.zeros(2, 6, 8) for _ in range(3))
        )
        random_visual = action_expert.prepare_layerwise_mot_visual_tokens(
            tuple(torch.randn(2, 6, 8) for _ in range(3))
        )
        masked_zero = action_expert.forward_layerwise_mot(
            action, timestep, context, zero_visual,
            visual_mask=torch.zeros(2, 6, dtype=torch.bool),
        )
        masked_random = action_expert.forward_layerwise_mot(
            action, timestep, context, random_visual,
            visual_mask=torch.zeros(2, 6, dtype=torch.bool),
        )
        unmasked_random = action_expert.forward_layerwise_mot(
            action, timestep, context, random_visual,
        )
    assert torch.allclose(masked_zero, masked_random, atol=1.0e-6, rtol=1.0e-6)
    assert not torch.allclose(masked_zero, unmasked_random)


def test_action_dit_uniform_pretrained_depth_mapping(
    path="/tmp/qwen_action_dit_uniform_mapping.pt",
):
    config = {
        "hidden_dim": 8,
        "action_dim": 2,
        "ffn_dim": 16,
        "text_dim": 8,
        "freq_dim": 8,
        "eps": 1.0e-6,
        "num_heads": 1,
        "attn_head_dim": 2,
        "num_layers": 5,
        "use_gradient_checkpointing": False,
    }
    source = ActionDiT(**config)
    with torch.no_grad():
        for layer_idx, block in enumerate(source.blocks):
            block.modulation.fill_(float(layer_idx + 1))
    backbone_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if not any(key.startswith(prefix) for prefix in ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES)
    }
    meta = {
        key: config[key]
        for key in ActionDiT.ACTION_BACKBONE_META_KEYS
    }
    torch.save({"policy": {}, "meta": meta, "backbone_state_dict": backbone_state}, path)

    target_config = dict(config)
    target_config["num_layers"] = 3
    target = ActionDiT.from_pretrained(
        action_dit_config=target_config,
        action_dit_pretrained_path=path,
        pretrained_layer_mapping="uniform",
        device="cpu",
        torch_dtype=torch.float32,
    )
    # Endpoint-aligned 5 -> 3 mapping is source blocks [0, 2, 4].
    assert [float(block.modulation.mean()) for block in target.blocks] == [1.0, 3.0, 5.0]


def test_exact_teacher_early_exit_matches_full_layer_capture():
    full_backbone = FakeBackbone()
    early_exit_backbone = copy.deepcopy(full_backbone)
    model_inputs = {"input_ids": torch.tensor([[0, 1, 2]])}

    _, full_layer2 = forward_with_decoder_layer_capture(
        full_backbone,
        layer=2,
        model_inputs=model_inputs,
    )
    original_depth = truncate_decoder_for_exact_early_exit(
        early_exit_backbone,
        layer=2,
    )
    early_output, early_layer2 = forward_with_decoder_layer_capture(
        early_exit_backbone,
        layer=2,
        model_inputs=model_inputs,
    )

    assert original_depth == 3
    assert len(early_exit_backbone.language_model.layers) == 2
    assert torch.equal(early_layer2, full_layer2)
    # The pruned backbone's final output now follows layer 2 directly, proving
    # that layer 3 was not executed.
    assert torch.equal(
        early_output.last_hidden_state.squeeze(-1),
        torch.tensor([[13.0, 14.0, 15.0]]),
    )


def test_visual_token_grouping():
    image_token_id = 99
    input_ids = torch.tensor(
        [
            [1, 99, 99, 2, 99, 99, 0],
            [1, 99, 99, 2, 99, 99, 3],
        ]
    )
    hidden = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    grouped = extract_grouped_visual_tokens(
        hidden,
        input_ids,
        image_token_id=image_token_id,
        num_images_per_sample=2,
        visual_tokens_per_image=2,
    )
    assert grouped.shape == (2, 2, 2, 3)
    assert torch.equal(grouped[0, 0, 0], hidden[0, 1])
    assert torch.equal(grouped[0, 1, 1], hidden[0, 5])


class ToyDecoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, **_kwargs):
        return hidden_states + torch.tanh(self.proj(hidden_states))


class ToyLanguageModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyDecoderLayer(hidden_dim) for _ in range(3)])


class ToyQwenBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=hidden_dim,
                use_cache=True,
                num_hidden_layers=3,
            ),
            vision_config=SimpleNamespace(patch_size=2, spatial_merge_size=1),
            image_token_id=99,
        )
        self.embed = nn.Embedding(100, hidden_dim)
        self.language_model = ToyLanguageModel(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed(input_ids)
        for layer in self.language_model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class ToyProcessor:
    image_token = "<|image_pad|>"

    def __init__(self, visual_tokens_per_image: int = 4):
        self.visual_tokens_per_image = visual_tokens_per_image

    def __call__(self, *, images, text, **_kwargs):
        image_counts = [item.count(self.image_token) for item in text]
        assert sum(image_counts) == len(images)
        sequences = []
        for image_count in image_counts:
            sequence = [1]
            for _ in range(image_count):
                sequence.extend([99] * self.visual_tokens_per_image)
                sequence.append(2)
            sequences.append(sequence)
        max_len = max(map(len, sequences))
        input_ids = torch.zeros((len(sequences), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for batch_idx, sequence in enumerate(sequences):
            input_ids[batch_idx, : len(sequence)] = torch.tensor(sequence)
            attention_mask[batch_idx, : len(sequence)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.zeros((len(images), 3, 4, 4)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
            "mm_token_type_ids": input_ids.eq(99).to(dtype=torch.int32),
        }


def test_toy_end_to_end_and_teacher_freeze(tmp_path="/tmp/qwen_fastwam_cpu_test.pt"):
    torch.manual_seed(0)
    student = ToyQwenBackbone()
    teacher = ToyQwenBackbone()
    action_expert = ActionDiT(
        hidden_dim=8,
        action_dim=2,
        ffn_dim=16,
        text_dim=8,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=1,
        attn_head_dim=2,
        num_layers=3,
        use_gradient_checkpointing=False,
    )
    model = JAMQwen35(
        student_backbone=student,
        teacher_backbone=teacher,
        processor=ToyProcessor(),
        action_expert=action_expert,
        model_id="toy-student",
        teacher_model_id="toy-teacher",
        proprio_dim=3,
        image_size=4,
        num_views=2,
        context_dim=8,
        predictor_hidden_dim=10,
        action_fusion_type=QWEN_ACTION_FUSION_LAYERWISE_MOT,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=20,
        loss_lambda_action=1.0,
        loss_lambda_foresight=0.2,
        foresight_cfg={
            "enabled": True,
            "load_teacher": True,
            "layer": 2,
            "frame_offsets": [4, 8],
            "offset_weights": None,
        },
        device="cpu",
        torch_dtype=torch.float32,
    )
    assert "teacher" not in dict(model.named_modules())
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert len(model.teacher_backbone.language_model.layers) == 2

    batch_size = 2
    sample = {
        "video": torch.rand((batch_size, 3, 9, 4, 8)) * 2.0 - 1.0,
        "action": torch.randn((batch_size, 4, 2)),
        "proprio": torch.randn((batch_size, 5, 3)),
        "prompt": ["pick up the cup", "open the drawer"],
        "image_is_pad": torch.tensor(
            [[False] * 9, [False, False, False, False, False, False, False, False, True]]
        ),
        "action_is_pad": torch.tensor(
            [[False, False, False, False], [False, False, True, True]]
        ),
    }
    loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())

    model.save_checkpoint(tmp_path, step=7)
    payload = model.load_checkpoint(tmp_path)
    assert payload["future_frame_offsets"] == (4, 8)
    assert payload["foresight_layer"] == 2
    assert payload["action_fusion_type"] == QWEN_ACTION_FUSION_LAYERWISE_MOT
    assert payload["qwen_num_layers"] == payload["action_num_layers"] == 3


def main():
    test_offset_layout()
    test_future_pad_mask()
    test_one_based_layer_capture()
    test_action_dit_layerwise_mot_mixes_and_masks_visual_tokens()
    test_action_dit_uniform_pretrained_depth_mapping()
    test_exact_teacher_early_exit_matches_full_layer_capture()
    test_visual_token_grouping()
    test_toy_end_to_end_and_teacher_freeze()
    print("QWEN FORESIGHT OFFSET/LAYER CHECKS PASSED")


if __name__ == "__main__":
    main()
