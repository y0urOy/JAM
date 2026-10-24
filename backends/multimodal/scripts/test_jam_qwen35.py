#!/usr/bin/env python
import copy
from collections import OrderedDict
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from jam_multimodal.models.backbones.jam_qwen35 import (
    QWEN_ACTION_FUSION_FINAL,
    QWEN_ACTION_FUSION_LAYERWISE_MOT,
    QWEN_ACTION_FUSION_NATIVE_QWEN,
    QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT,
    QWEN_WAN_CHECKPOINT_FORMAT,
    QWEN_WAN_LEGACY_CHECKPOINT_FORMAT,
    QWEN_JAM_CHECKPOINT_FORMAT,
    QWEN_PREDICTOR_JAM,
    QWEN_PREDICTOR_LEGACY,
    JAMQwen35,
    build_random_qwen_backbone,
    extract_grouped_visual_tokens,
    forward_with_decoder_layer_capture,
    future_target_valid_mask,
    resolve_decoded_frame_positions,
    resolve_teacher_frame_layout,
    truncate_decoder_for_exact_early_exit,
    uniform_endpoint_layer_map,
)
from jam_multimodal.models.backbones.action_dit import ActionDiT
from jam_multimodal.models.backbones.action_dit_yak import ActionDiTYak
from jam_multimodal.models.backbones.foresight_yak import ForesightProjector


def test_offset_layout():
    frames, targets = resolve_teacher_frame_layout([4])
    assert frames == (0, 4)
    assert targets == (1,)

    frames, targets = resolve_teacher_frame_layout([0, 4])
    assert frames == (0, 4)
    assert targets == (0, 1)


def test_future_pad_mask():
    image_is_pad = torch.tensor(
        [
            [False, False, False, False, False, False, False, False, True],
            [False, False, False, False, True, True, True, True, True],
        ]
    )
    valid = future_target_valid_mask(image_is_pad, [4])
    expected = torch.tensor([[True], [False]])
    assert torch.equal(valid, expected)

    no_pad = future_target_valid_mask(None, [4], batch_size=2, device=torch.device("cpu"))
    assert torch.equal(no_pad, torch.ones((2, 1), dtype=torch.bool))

    endpoint_pad = torch.tensor([[False, False], [False, True]])
    endpoint_valid = future_target_valid_mask(endpoint_pad, [1])
    assert torch.equal(endpoint_valid, torch.tensor([[True], [False]]))


def test_decoded_frame_positions():
    assert resolve_decoded_frame_positions((0, 4), 5) == (0, 4)
    assert resolve_decoded_frame_positions((0, 4), 2) == (0, 1)
    assert resolve_decoded_frame_positions((0, 16), 2) == (0, 1)
    try:
        resolve_decoded_frame_positions((0, 4, 8), 2)
    except ValueError as exc:
        assert "cannot represent" in str(exc)
    else:
        raise AssertionError("Multi-future offsets must reject endpoint-only input.")


def test_batched_uint8_conversion_matches_legacy_per_view_path():
    torch.manual_seed(29)
    views = torch.rand((3, 6, 3, 7, 11), dtype=torch.float32) * 2.2 - 1.1
    expected = torch.stack(
        [
            torch.stack(
                [JAMQwen35._to_uint8_image(view) for view in sample], dim=0
            )
            for sample in views
        ],
        dim=0,
    )
    actual = JAMQwen35._to_uint8_images(views)
    assert actual.dtype == torch.uint8
    assert torch.equal(actual, expected)


def test_horizontal_camera_split_is_exact_zero_copy_view():
    model = JAMQwen35.__new__(JAMQwen35)
    nn.Module.__init__(model)
    model.image_size = 4
    model.num_views = 3
    frame = torch.arange(2 * 3 * 4 * 12, dtype=torch.float32).reshape(
        2, 3, 4, 12
    )
    expected = torch.stack(frame.split(4, dim=-1), dim=1)
    actual = model._split_views(frame)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().data_ptr() == frame.untyped_storage().data_ptr()


def test_tensor_native_qwen_preprocessing_matches_official_processor():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "checkpoints/Qwen3.5-2B-Base",
        min_pixels=224 * 224,
        max_pixels=224 * 224,
        local_files_only=True,
    )
    model = JAMQwen35.__new__(JAMQwen35)
    nn.Module.__init__(model)
    model.processor = processor
    model.image_size = 224
    model.visual_tokens_per_view = 49
    model.image_token_id = int(processor.image_token_id)
    model.torch_dtype = torch.bfloat16
    model.device = torch.device("cpu")
    model._tokenized_cache_max_entries = 8
    model._tokenized_layout_cache = OrderedDict()

    torch.manual_seed(31)
    views = torch.rand((2, 3, 3, 224, 224), dtype=torch.float32) * 2.2 - 1.1
    placeholder = model._vision_placeholder()
    texts = [
        f"move the red cube\n{placeholder}\n{placeholder}\n{placeholder}",
        f"put object into container\n{placeholder}\n{placeholder}\n{placeholder}",
    ]
    model.tensor_native_preprocessing = False
    expected = model._prepare_qwen_inputs(views, texts)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    rope_reference = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        )
    )
    rope_reference.get_vision_position_ids = MethodType(
        Qwen3_5Model.get_vision_position_ids,
        rope_reference,
    )
    expected_position_ids, _ = Qwen3_5Model.get_rope_index(
        rope_reference,
        expected["input_ids"],
        expected["mm_token_type_ids"],
        image_grid_thw=expected["image_grid_thw"],
        attention_mask=expected["attention_mask"],
    )
    model.tensor_native_preprocessing = True
    actual = model._prepare_qwen_inputs(views, texts)
    assert actual.keys() == expected.keys() | {
        "position_ids",
        "_image_token_indices",
    }
    for key in expected:
        assert torch.equal(expected[key], actual[key]), (
            key,
            expected[key].shape,
            actual[key].shape,
            (expected[key].float() - actual[key].float()).abs().max(),
        )
    assert torch.equal(expected_position_ids, actual["position_ids"])
    expected_image_indices = torch.stack(
        [
            torch.nonzero(
                row.eq(model.image_token_id), as_tuple=False
            ).flatten()
            for row in expected["input_ids"]
        ],
        dim=0,
    )
    assert torch.equal(
        expected_image_indices,
        actual["_image_token_indices"],
    )
    cache_entries = len(model._tokenized_layout_cache)
    repeated = model._prepare_qwen_inputs(views, texts)
    assert len(model._tokenized_layout_cache) == cache_entries
    for key in actual:
        assert torch.equal(actual[key], repeated[key]), key

    prepared_pixels = model._prepare_qwen_pixels_tensor_native(views)
    prepared = model._prepare_qwen_inputs(
        views,
        texts,
        prepared_pixels=prepared_pixels,
    )
    for key in actual:
        assert torch.equal(actual[key], prepared[key]), key

    # Packing current+future once and selecting the leading current views must
    # be bit-identical to packing the current views independently.
    future_views = torch.flip(views, dims=(-1,))
    teacher_views = torch.cat((views, future_views), dim=1)
    teacher_pixels, teacher_grid = model._prepare_qwen_pixels_tensor_native(
        teacher_views
    )
    rows_per_image = teacher_pixels.shape[0] // (views.shape[0] * 6)
    reused_current_pixels = teacher_pixels.reshape(
        views.shape[0], 6, rows_per_image, teacher_pixels.shape[-1]
    )[:, :3].reshape(-1, teacher_pixels.shape[-1])
    reused_current_grid = teacher_grid.reshape(views.shape[0], 6, 3)[:, :3].reshape(
        -1, 3
    )
    assert torch.equal(prepared_pixels[0], reused_current_pixels)
    assert torch.equal(prepared_pixels[1], reused_current_grid)

    # More unique rows than the configured bound must still produce a valid
    # current batch while evicting old GPU/CPU entries afterwards.
    model._tokenized_cache_max_entries = 1
    other_texts = [
        f"new task {idx}\n{placeholder}\n{placeholder}\n{placeholder}"
        for idx in range(2)
    ]
    bounded = model._prepare_qwen_inputs(views, other_texts)
    assert bounded["input_ids"].shape[0] == 2
    assert len(model._tokenized_layout_cache) == 1


def test_native_qwen_carries_only_live_proprio_context():
    class CaptureAction(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def prepare_condition(self, **kwargs):
            self.received = kwargs
            return kwargs

    model = JAMQwen35.__new__(JAMQwen35)
    nn.Module.__init__(model)
    model.proprio_dim = 3
    model.context_dim = 8
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.action_fusion_type = QWEN_ACTION_FUSION_NATIVE_QWEN
    capture = CaptureAction()
    model.dit = nn.ModuleDict(
        {
            "proprio_projector": nn.Sequential(
                nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 8)
            ),
            "action_expert": capture,
        }
    )
    context = torch.randn(2, 7, 8)
    context_mask = torch.ones(2, 7, dtype=torch.bool)
    proprio = torch.randn(2, 5, 3)
    live_context, live_mask, layerwise = model._append_proprio_context(
        context,
        context_mask,
        proprio,
    )
    assert live_context.shape == (2, 1, 8)
    assert live_mask.shape == (2, 1)
    assert layerwise is None
    native_inputs = {
        "layerwise_prefix_tokens": (),
        "prefix_attention_mask": context_mask,
        "prefix_position_ids": torch.arange(7)[None, None].expand(3, 2, -1),
    }
    prepared = model._prepare_action_prefix(
        context=live_context,
        context_mask=live_mask,
        layerwise_visual_tokens=None,
        layerwise_text_tokens=None,
        native_action_inputs=native_inputs,
    )
    assert prepared["proprio_token"] is live_context
    assert capture.received is prepared


class FakeConditionalGeneration:
    last_kwargs = None

    @classmethod
    def _from_config(cls, config, **kwargs):
        cls.last_kwargs = kwargs
        return SimpleNamespace(model=nn.Linear(config.width, config.width))


def test_random_teacher_builder_is_deterministic_and_rng_isolated():
    config = SimpleNamespace(width=4)
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()
    first = build_random_qwen_backbone(
        FakeConditionalGeneration,
        config,
        seed=42,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    assert torch.equal(state_before, torch.random.get_rng_state())
    second = build_random_qwen_backbone(
        FakeConditionalGeneration,
        config,
        seed=42,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    third = build_random_qwen_backbone(
        FakeConditionalGeneration,
        config,
        seed=43,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(first.parameters(), second.parameters())
    )
    assert any(
        not torch.equal(first_value, third_value)
        for first_value, third_value in zip(first.parameters(), third.parameters())
    )
    assert FakeConditionalGeneration.last_kwargs == {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }


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


def test_qwen_yak18_layer_map_and_direct_prefix_backward():
    assert uniform_endpoint_layer_map(24, 18) == (
        0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23
    )
    torch.manual_seed(29)
    action_expert = ActionDiTYak(
        action_dim=2,
        residual_dim=128,
        num_heads=1,
        num_layers_double=1,
        num_layers_single=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        max_action_horizon=8,
    )
    text_layers = tuple(
        torch.randn(2, 5, 128, requires_grad=True) for _ in range(2)
    )
    visual_layers = tuple(
        torch.randn(2, 8, 128, requires_grad=True) for _ in range(2)
    )
    text_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    prefix = action_expert.prepare_qwen_layerwise_prefix(
        layerwise_text_tokens=text_layers,
        layerwise_visual_tokens=visual_layers,
        text_mask=text_mask,
        visual_grid_height=2,
        visual_grid_width=2,
        num_views=2,
    )
    action = torch.randn(2, 4, 2, requires_grad=True)
    output = action_expert.forward_qwen_layerwise_mot(
        action_tokens=action,
        timestep=torch.tensor([0.2, 0.7]),
        prefix_cache=prefix,
    )
    assert output.shape == action.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert all(item.grad is not None for item in text_layers)
    assert all(item.grad is not None for item in visual_layers)
    assert QWEN_ACTION_FUSION_YAK_LAYERWISE_MOT == "yak_layerwise_mot"


def test_action_dit_checkpointing_matches_saved_activation_backward():
    """The custom ActionDiT checkpoint path must preserve outputs and gradients."""

    torch.manual_seed(19)
    checkpointed = ActionDiT(
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
    uncheckpointed = copy.deepcopy(checkpointed)
    uncheckpointed.use_gradient_checkpointing = False
    checkpointed.train()
    uncheckpointed.train()

    action_seed = torch.randn(2, 4, 2)
    timestep = torch.rand(2)
    context_seed = torch.randn(2, 5, 8)
    visual_seed = tuple(torch.randn(2, 6, 8) for _ in range(3))

    def run(model):
        action = action_seed.detach().clone().requires_grad_(True)
        context = context_seed.detach().clone().requires_grad_(True)
        visual_contexts = tuple(
            item.detach().clone().requires_grad_(True) for item in visual_seed
        )
        visual_tokens = model.prepare_layerwise_mot_visual_tokens(visual_contexts)
        output = model.forward_layerwise_mot(
            action_tokens=action,
            timestep=timestep,
            context=context,
            layerwise_visual_tokens=visual_tokens,
        )
        output.square().mean().backward()
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return (
            output.detach(),
            action.grad.detach(),
            context.grad.detach(),
            tuple(item.grad.detach() for item in visual_contexts),
            parameter_grads,
        )

    checkpointed_result = run(checkpointed)
    uncheckpointed_result = run(uncheckpointed)
    for checkpointed_tensor, uncheckpointed_tensor in zip(
        checkpointed_result[:3], uncheckpointed_result[:3]
    ):
        assert torch.allclose(
            checkpointed_tensor, uncheckpointed_tensor, atol=1.0e-6, rtol=1.0e-6
        )
    for checkpointed_tensor, uncheckpointed_tensor in zip(
        checkpointed_result[3], uncheckpointed_result[3]
    ):
        assert torch.allclose(
            checkpointed_tensor, uncheckpointed_tensor, atol=1.0e-6, rtol=1.0e-6
        )
    assert checkpointed_result[4].keys() == uncheckpointed_result[4].keys()
    for name in checkpointed_result[4]:
        assert torch.allclose(
            checkpointed_result[4][name],
            uncheckpointed_result[4][name],
            atol=1.0e-6,
            rtol=1.0e-6,
        ), name


def test_action_dit_uniform_pretrained_depth_mapping(
    path="/tmp/qwen_jam_action_dit_uniform_mapping.pt",
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
    meta = {key: config[key] for key in ActionDiT.ACTION_BACKBONE_META_KEYS}
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
    hidden = (
        torch.arange(2 * 7 * 3, dtype=torch.float32)
        .reshape(2, 7, 3)
        .requires_grad_(True)
    )
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
    grouped.square().sum().backward()
    expected_grad = torch.zeros_like(hidden)
    image_mask = input_ids.eq(image_token_id)
    expected_grad[image_mask] = 2.0 * hidden.detach()[image_mask]
    assert torch.equal(hidden.grad, expected_grad)


def test_jam_projector_mixes_tokens():
    torch.manual_seed(11)
    projector = ForesightProjector(
        in_dim=6,
        hidden_dim=4,
        ffn_dim=8,
        num_layers=2,
        num_heads=2,
        num_targets=1,
        out_dim=6,
    )
    baseline = torch.zeros(1, 3, 6)
    perturbed = baseline.clone()
    perturbed[:, 0] = 1.0
    baseline_out = projector.head(projector.trunk(baseline), 0)
    perturbed_out = projector.head(projector.trunk(perturbed), 0)
    # Changing token 0 affects token 1 only if the predictor mixes tokens.
    assert not torch.equal(baseline_out[:, 1], perturbed_out[:, 1])


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


def _make_toy_qwen_model(
    predictor_type=QWEN_PREDICTOR_JAM,
    action_fusion_type=QWEN_ACTION_FUSION_LAYERWISE_MOT,
    teacher_mode="frozen_pretrained",
    teacher_random_seed=42,
    num_views=2,
    frame_offsets=(4,),
):
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
        num_layers=3 if action_fusion_type == QWEN_ACTION_FUSION_LAYERWISE_MOT else 2,
        use_gradient_checkpointing=False,
    )
    imagewam_predictor = predictor_type == QWEN_PREDICTOR_JAM
    model = JAMQwen35(
        student_backbone=student,
        teacher_backbone=teacher,
        processor=ToyProcessor(),
        action_expert=action_expert,
        model_id="toy-student",
        teacher_model_id="toy-teacher",
        proprio_dim=3,
        image_size=4,
        num_views=num_views,
        context_dim=8,
        predictor_type=predictor_type,
        predictor_hidden_dim=4 if imagewam_predictor else 10,
        predictor_ffn_dim=8,
        predictor_num_layers=2,
        predictor_num_heads=2,
        action_fusion_type=action_fusion_type,
        action_train_shift=5.0,
        action_infer_shift=5.0,
        action_num_train_timesteps=20,
        loss_lambda_action=1.0,
        loss_lambda_foresight=0.2,
        foresight_cfg={
            "enabled": True,
            "load_teacher": True,
            "teacher_mode": teacher_mode,
            "teacher_random_seed": teacher_random_seed,
            "layer": 2,
            "frame_offsets": list(frame_offsets),
            "offset_weights": None,
        },
        device="cpu",
        torch_dtype=torch.float32,
    )
    return model, student, teacher


def test_robotwin_three_view_endpoint_training_loss():
    """Exercise the exact RoboTwin view/endpoint/action geometry on the toy model."""

    torch.manual_seed(31)
    model, _, teacher = _make_toy_qwen_model(num_views=3, frame_offsets=(16,))
    sample = {
        "video": torch.rand((2, 3, 2, 4, 12)) * 2.0 - 1.0,
        "action": torch.randn((2, 16, 2)),
        "proprio": torch.randn((2, 17, 3)),
        "prompt": ["put the bottle in the bin", "open the microwave"],
        "image_is_pad": torch.tensor([[False, False], [False, True]]),
        "action_is_pad": torch.tensor(
            [[False] * 16, [False] * 12 + [True] * 4]
        ),
    }
    assert model._validate_training_sample(sample) == (0, 1)
    loss, metrics = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.as_tensor(value)) for value in metrics.values())
    loss.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert model.num_views == 3
    assert model.future_frame_offsets == (16,)


def test_teacher_first_is_loss_and_gradient_exact():
    """Reordering the frozen teacher must not alter any trainable gradient."""

    torch.manual_seed(43)
    model, _, teacher = _make_toy_qwen_model(num_views=3, frame_offsets=(16,))
    sample = {
        "video": torch.rand((2, 3, 2, 4, 12)) * 2.0 - 1.0,
        "action": torch.randn((2, 16, 2)),
        "proprio": torch.randn((2, 17, 3)),
        "prompt": ["move the mug", "close the drawer"],
        "image_is_pad": torch.tensor([[False, False], [False, True]]),
        "action_is_pad": torch.tensor(
            [[False] * 16, [False] * 13 + [True] * 3]
        ),
    }

    def run(teacher_first: bool):
        model.zero_grad(set_to_none=True)
        model.teacher_first = teacher_first
        # Student dropout/action-noise streams must start from the same state.
        # The frozen eval-mode teacher is RNG-free, so both execution orders
        # should consume those streams identically.
        torch.manual_seed(20260903)
        loss, metrics = model.training_loss(sample)
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        return loss.detach().clone(), {
            key: value.detach().clone() for key, value in metrics.items()
        }, grads

    legacy_loss, legacy_metrics, legacy_grads = run(False)
    reordered_loss, reordered_metrics, reordered_grads = run(True)
    assert torch.equal(legacy_loss, reordered_loss)
    assert legacy_metrics.keys() == reordered_metrics.keys()
    for key in legacy_metrics:
        assert torch.equal(legacy_metrics[key], reordered_metrics[key]), key
    assert legacy_grads.keys() == reordered_grads.keys()
    for name in legacy_grads:
        assert torch.equal(legacy_grads[name], reordered_grads[name]), name
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_toy_end_to_end_and_teacher_freeze(tmp_path="/tmp/qwen_jam_cpu_test.pt"):
    torch.manual_seed(0)
    model, student, teacher = _make_toy_qwen_model()
    assert "teacher" not in dict(model.named_modules())
    assert "foresight_projector" in model.dit
    assert "foresight_predictors" not in model.dit
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
    assert all(torch.isfinite(torch.as_tensor(value)) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        parameter.grad is not None
        for parameter in model.foresight_projector.parameters()
    )

    endpoint_sample = dict(sample)
    endpoint_sample["video"] = sample["video"][:, :, [0, 4]]
    endpoint_sample["image_is_pad"] = sample["image_is_pad"][:, [0, 4]]
    assert model._validate_training_sample(endpoint_sample) == (0, 1)
    endpoint_loss, endpoint_metrics = model.training_loss(endpoint_sample)
    assert torch.isfinite(endpoint_loss)
    assert all(torch.isfinite(torch.as_tensor(value)) for value in endpoint_metrics.values())

    model.save_checkpoint(tmp_path, step=7)
    payload = model.load_checkpoint(tmp_path)
    assert payload["format"] == QWEN_JAM_CHECKPOINT_FORMAT
    assert payload["future_frame_offsets"] == (4,)
    assert payload["foresight_layer"] == 2
    assert payload["loss_lambda_foresight"] == 0.2
    assert payload["foresight_predictor_type"] == QWEN_PREDICTOR_JAM
    assert payload["action_fusion_type"] == QWEN_ACTION_FUSION_LAYERWISE_MOT
    assert payload["qwen_num_layers"] == payload["action_num_layers"] == 3
    assert payload["teacher_mode"] == "frozen_pretrained"
    assert payload["teacher_random_seed"] == 42
    assert payload["foresight_predictor_config"] == {
        "hidden_dim": 4,
        "ffn_dim": 8,
        "num_layers": 2,
        "num_heads": 2,
    }


def test_legacy_fastwam_checkpoint_remains_loadable(
    tmp_path="/tmp/qwen_jam_legacy_cpu_test.pt",
):
    torch.manual_seed(3)
    model, _, _ = _make_toy_qwen_model(
        QWEN_PREDICTOR_LEGACY,
        action_fusion_type=QWEN_ACTION_FUSION_FINAL,
    )
    model.save_checkpoint(tmp_path, step=5)
    payload = torch.load(tmp_path, map_location="cpu")
    assert payload["format"] == QWEN_WAN_CHECKPOINT_FORMAT
    # Emulate an old FastWAM-Qwen payload, which predates explicit predictor and
    # action-fusion metadata. The v1 format implies the legacy paths.
    payload["format"] = QWEN_WAN_LEGACY_CHECKPOINT_FORMAT
    payload.pop("foresight_predictor_type")
    payload.pop("foresight_predictor_config")
    payload.pop("action_fusion_type")
    payload.pop("qwen_num_layers")
    payload.pop("action_num_layers")
    payload.pop("teacher_mode")
    payload.pop("teacher_random_seed")
    torch.save(payload, tmp_path)
    loaded = model.load_checkpoint(tmp_path)
    assert loaded["format"] == QWEN_WAN_LEGACY_CHECKPOINT_FORMAT


def test_random_teacher_checkpoint_identity(
    tmp_path="/tmp/qwen_jam_random_teacher_cpu_test.pt",
):
    model, _, _ = _make_toy_qwen_model(
        teacher_mode="frozen_random",
        teacher_random_seed=17,
    )
    model.save_checkpoint(tmp_path, step=1)
    payload = model.load_checkpoint(tmp_path)
    assert payload["teacher_mode"] == "frozen_random"
    assert payload["teacher_random_seed"] == 17

    mismatched, _, _ = _make_toy_qwen_model(
        teacher_mode="frozen_random",
        teacher_random_seed=18,
    )
    try:
        mismatched.load_checkpoint(tmp_path)
    except ValueError as exc:
        assert "random-teacher seed mismatch" in str(exc)
    else:
        raise AssertionError("Random-teacher checkpoint must reject a different seed.")


def main():
    test_offset_layout()
    test_future_pad_mask()
    test_decoded_frame_positions()
    test_batched_uint8_conversion_matches_legacy_per_view_path()
    test_horizontal_camera_split_is_exact_zero_copy_view()
    test_tensor_native_qwen_preprocessing_matches_official_processor()
    test_native_qwen_carries_only_live_proprio_context()
    test_random_teacher_builder_is_deterministic_and_rng_isolated()
    test_one_based_layer_capture()
    test_action_dit_layerwise_mot_mixes_and_masks_visual_tokens()
    test_qwen_yak18_layer_map_and_direct_prefix_backward()
    test_action_dit_checkpointing_matches_saved_activation_backward()
    test_action_dit_uniform_pretrained_depth_mapping()
    test_exact_teacher_early_exit_matches_full_layer_capture()
    test_visual_token_grouping()
    test_jam_projector_mixes_tokens()
    test_robotwin_three_view_endpoint_training_loss()
    test_teacher_first_is_loss_and_gradient_exact()
    test_toy_end_to_end_and_teacher_freeze()
    test_legacy_fastwam_checkpoint_remains_loadable()
    test_random_teacher_checkpoint_identity()
    print("QWEN FORESIGHT OFFSET/LAYER CHECKS PASSED")


if __name__ == "__main__":
    main()
