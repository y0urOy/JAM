from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


QWEN_ACTION_ATTENTION_CAUSAL = "causal"
QWEN_ACTION_ATTENTION_MODES = {
    QWEN_ACTION_ATTENTION_CAUSAL,
}


@dataclass(frozen=True)
class Qwen35ActionCondition:
    """Layerwise Qwen prefix states consumed by the native action stream.

    The prefix tensors are outputs of the trainable policy Qwen decoder layers.
    They remain attached to the student graph, so action loss trains the visual
    policy.  The frozen teacher is never stored here and remains training-only.
    """

    layerwise_prefix_tokens: tuple[torch.Tensor, ...]
    prefix_attention_mask: torch.Tensor
    prefix_position_ids: torch.Tensor
    proprio_token: Optional[torch.Tensor] = None


class ActionDiTQwen35Native(nn.Module):
    """Qwen3.5-initialized action expert with a native 24-layer block pattern.

    Every action block is an independent copy of the matching pretrained Qwen
    decoder block.  Consequently the action stream preserves Qwen3.5-2B's
    ``[linear, linear, linear, full] x 6`` token-mixer pattern, RMSNorms, MLPs,
    attention geometry, and GatedDeltaNet parameters at initialization.

    At layer ``i`` the corresponding trainable policy-Qwen prefix is placed
    before the noisy action tokens.  The visual/text prefix never reads action
    tokens.  The ``causal`` action-suffix mode exactly preserves Qwen's decoder
    mask, so an action token reads the complete prefix and earlier actions.

    Action input, action position, diffusion-time embedding, proprio projection
    (owned by the policy wrapper), and action output head are action-specific and
    newly initialized, matching the initialization convention of the native
    DINO action ports.
    """

    FAMILY = "ActionDiTQwen35Native24"

    def __init__(
        self,
        *,
        blocks: Sequence[nn.Module],
        rotary_emb: nn.Module,
        final_norm: nn.Module,
        text_config,
        action_dim: int = 7,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        initializer_range: Optional[float] = None,
        use_gradient_checkpointing: bool = False,
        action_attention_mode: str = QWEN_ACTION_ATTENTION_CAUSAL,
        skip_discarded_prefix_mlp: bool = True,
        skip_discarded_prefix_query: bool = False,
    ) -> None:
        super().__init__()
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("Qwen native action expert requires at least one block.")

        self.hidden_dim = int(text_config.hidden_size)
        self.action_dim = int(action_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.initializer_range = float(
            getattr(text_config, "initializer_range", 0.02)
            if initializer_range is None
            else initializer_range
        )
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        # Each action layer receives a fresh policy-Qwen prefix.  The prefix
        # output produced by this action block is therefore discarded before
        # the next layer; only the action rows survive.  LayerNorm and the MLP
        # are token-wise, so running that subgraph on action rows only is
        # mathematically identical and avoids a large amount of dead work.
        self.skip_discarded_prefix_mlp = bool(skip_discarded_prefix_mlp)
        # In a GatedDeltaNet block, prefix K/V, beta and decay update the
        # recurrent state read by action rows, but prefix query values affect
        # only prefix outputs that are discarded.  Query convolution has a
        # short left context, so an optimized path need project Q only for the
        # action suffix plus ``kernel_size - 1`` preceding rows.
        self.skip_discarded_prefix_query = bool(
            skip_discarded_prefix_query
        )
        self.action_attention_mode = str(action_attention_mode).strip().lower()
        if self.action_attention_mode not in QWEN_ACTION_ATTENTION_MODES:
            raise ValueError(
                "Qwen native action_attention_mode must be one of "
                f"{sorted(QWEN_ACTION_ATTENTION_MODES)}, got "
                f"{action_attention_mode!r}."
            )
        self.layer_types = tuple(str(kind) for kind in text_config.layer_types)
        self.num_layers = len(blocks)

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.max_action_horizon <= 0:
            raise ValueError(
                "max_action_horizon must be positive, "
                f"got {self.max_action_horizon}."
            )
        if self.timestep_embed_dim < 2:
            raise ValueError(
                "timestep_embed_dim must be at least 2, "
                f"got {self.timestep_embed_dim}."
            )
        if self.num_layers != len(self.layer_types):
            raise ValueError(
                "Qwen action block depth must match text_config.layer_types: "
                f"blocks={self.num_layers}, layer_types={len(self.layer_types)}."
            )
        actual_types = tuple(str(getattr(block, "block_type", "")) for block in blocks)
        if actual_types != self.layer_types:
            raise ValueError(
                "Qwen action block types do not match the source config: "
                f"blocks={actual_types}, config={self.layer_types}."
            )

        # These modules are already independent copies of the source Qwen
        # modules.  Registering them here must not reinitialize their weights.
        self.blocks = nn.ModuleList(blocks)
        self.rotary_emb = rotary_emb
        self.final_norm = final_norm

        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.action_position = nn.Parameter(
            torch.empty(1, self.max_action_horizon, self.hidden_dim)
        )
        self.time_in = nn.Sequential(
            nn.Linear(self.timestep_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.head = nn.Linear(self.hidden_dim, self.action_dim)
        self._reset_action_specific_parameters()

    @classmethod
    def from_qwen_backbone(
        cls,
        backbone: nn.Module,
        *,
        action_dim: int = 7,
        max_action_horizon: int = 64,
        timestep_embed_dim: int = 256,
        initializer_range: Optional[float] = None,
        use_gradient_checkpointing: bool = False,
        action_attention_mode: str = QWEN_ACTION_ATTENTION_CAUSAL,
        skip_discarded_prefix_mlp: bool = True,
        skip_discarded_prefix_query: bool = False,
        device: str | torch.device = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "ActionDiTQwen35Native":
        language_model = getattr(backbone, "language_model", None)
        source_blocks = getattr(language_model, "layers", None)
        source_rotary = getattr(language_model, "rotary_emb", None)
        source_norm = getattr(language_model, "norm", None)
        text_config = getattr(getattr(backbone, "config", None), "text_config", None)
        if not isinstance(source_blocks, nn.ModuleList):
            raise TypeError(
                "Expected Qwen backbone.language_model.layers to be nn.ModuleList, "
                f"got {type(source_blocks)}."
            )
        if source_rotary is None or source_norm is None or text_config is None:
            raise TypeError(
                "Qwen backbone must expose language_model.rotary_emb/norm and "
                "config.text_config."
            )

        # deepcopy is intentional: action and visual weights start equal
        # but must be independently trainable after initialization.
        blocks = [copy.deepcopy(block) for block in source_blocks]
        rotary_emb = copy.deepcopy(source_rotary)
        final_norm = copy.deepcopy(source_norm)
        action = cls(
            blocks=blocks,
            rotary_emb=rotary_emb,
            final_norm=final_norm,
            text_config=copy.deepcopy(text_config),
            action_dim=action_dim,
            max_action_horizon=max_action_horizon,
            timestep_embed_dim=timestep_embed_dim,
            initializer_range=initializer_range,
            use_gradient_checkpointing=use_gradient_checkpointing,
            action_attention_mode=action_attention_mode,
            skip_discarded_prefix_mlp=skip_discarded_prefix_mlp,
            skip_discarded_prefix_query=skip_discarded_prefix_query,
        )
        if torch_dtype is None:
            torch_dtype = next(backbone.parameters()).dtype
        return action.to(device=device, dtype=torch_dtype)

    def _reset_action_specific_parameters(self) -> None:
        std = self.initializer_range
        for module in (self.action_encoder, self.time_in[0], self.time_in[2], self.head):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.action_position, mean=0.0, std=std)

    @staticmethod
    def _assert_scalar(condition: torch.Tensor, message: str) -> None:
        """Validate a scalar tensor without synchronizing healthy CUDA steps."""

        if condition.numel() != 1:
            raise ValueError(
                f"ActionQwen assertion must be scalar, got {tuple(condition.shape)}."
            )
        if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
            torch._assert_async(condition, message)
        elif not bool(condition.item()):
            raise ValueError(message)

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dim: int,
        max_period: float = 10_000.0,
    ) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(
                f"timestep must be one-dimensional [B], got {tuple(timestep.shape)}."
            )
        half = int(dim) // 2
        if half <= 0:
            raise ValueError(f"timestep embedding dim must be >=2, got {dim}.")
        frequencies = torch.exp(
            -math.log(float(max_period))
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / float(half)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if int(dim) % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def prepare_condition(
        self,
        *,
        layerwise_prefix_tokens: Sequence[torch.Tensor],
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        proprio_token: Optional[torch.Tensor] = None,
    ) -> Qwen35ActionCondition:
        prefixes = tuple(layerwise_prefix_tokens)
        if len(prefixes) != self.num_layers:
            raise ValueError(
                "Qwen native action requires one policy prefix per action block: "
                f"prefixes={len(prefixes)}, blocks={self.num_layers}."
            )
        if prefix_attention_mask.ndim != 2:
            raise ValueError(
                "prefix_attention_mask must be [B,L], got "
                f"{tuple(prefix_attention_mask.shape)}."
            )
        batch_size, prefix_len = map(int, prefix_attention_mask.shape)
        if prefix_position_ids.ndim != 3 or tuple(prefix_position_ids.shape) != (
            3,
            batch_size,
            prefix_len,
        ):
            raise ValueError(
                "prefix_position_ids must be [3,B,L], got "
                f"{tuple(prefix_position_ids.shape)} for B/L={batch_size}/{prefix_len}."
            )
        for layer_idx, prefix in enumerate(prefixes):
            if tuple(prefix.shape) != (batch_size, prefix_len, self.hidden_dim):
                raise ValueError(
                    f"Qwen prefix layer {layer_idx} must be "
                    f"{(batch_size, prefix_len, self.hidden_dim)}, got {tuple(prefix.shape)}."
                )

        mask = prefix_attention_mask.to(dtype=torch.bool)
        lengths = mask.sum(dim=1)
        self._assert_scalar(
            (lengths > 0).all(),
            "Every Qwen action prefix must contain a valid token.",
        )
        expected_right_padding = (
            torch.arange(prefix_len, device=mask.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        self._assert_scalar(
            torch.all(mask == expected_right_padding),
            "Qwen native action currently requires right-padded contiguous prefixes.",
        )

        if proprio_token is not None:
            if tuple(proprio_token.shape) != (batch_size, 1, self.hidden_dim):
                raise ValueError(
                    "proprio_token must be [B,1,hidden], got "
                    f"{tuple(proprio_token.shape)}."
                )
            proprio_token = proprio_token.to(
                device=prefixes[0].device,
                dtype=prefixes[0].dtype,
            )

        return Qwen35ActionCondition(
            layerwise_prefix_tokens=prefixes,
            prefix_attention_mask=mask,
            prefix_position_ids=prefix_position_ids.to(
                device=prefixes[0].device,
                dtype=torch.long,
            ),
            proprio_token=proprio_token,
        )

    def _embed_action(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if action_tokens.ndim != 3 or int(action_tokens.shape[-1]) != self.action_dim:
            raise ValueError(
                f"action_tokens must be [B,A,{self.action_dim}], got {tuple(action_tokens.shape)}."
            )
        batch_size, action_len = map(int, action_tokens.shape[:2])
        if not 1 <= action_len <= self.max_action_horizon:
            raise ValueError(
                f"action length must be in [1,{self.max_action_horizon}], got {action_len}."
            )
        if timestep.ndim == 0:
            timestep = timestep[None]
        if timestep.ndim != 1:
            raise ValueError(
                f"timestep must be scalar or [B], got {tuple(timestep.shape)}."
            )
        if int(timestep.shape[0]) == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if int(timestep.shape[0]) != batch_size:
            raise ValueError(
                f"timestep batch={timestep.shape[0]} does not match action batch={batch_size}."
            )

        device = self.action_encoder.weight.device
        dtype = self.action_encoder.weight.dtype
        hidden = self.action_encoder(action_tokens.to(device=device, dtype=dtype))
        time_features = self.timestep_embedding(
            timestep.to(device=device), self.timestep_embed_dim
        )
        time_hidden = self.time_in(time_features.to(dtype=dtype))
        return (
            hidden
            + self.action_position[:, :action_len].to(device=device, dtype=dtype)
            + time_hidden[:, None, :]
        )

    @staticmethod
    def _build_full_causal_mask(
        valid_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len = map(int, valid_mask.shape)
        indices = torch.arange(seq_len, device=valid_mask.device)
        causal = indices[None, :] <= indices[:, None]
        allowed = causal[None, None] & valid_mask[:, None, None, :]
        query_valid = valid_mask[:, None, :, None]
        allowed = allowed & query_valid

        # Padding queries are discarded, but an all-masked SDPA row can become
        # NaN on some kernels. Give each padding query one harmless key.
        padding_queries = ~query_valid.squeeze(-1)
        allowed[:, :, :, 0] |= padding_queries
        return torch.zeros(
            (batch_size, 1, seq_len, seq_len),
            device=valid_mask.device,
            dtype=dtype,
        ).masked_fill(~allowed, torch.finfo(dtype).min)

    @staticmethod
    def _build_action_suffix_causal_mask(
        valid_mask: torch.Tensor,
        action_indices: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build exactly the action-query rows of the full causal mask."""

        seq_len = int(valid_mask.shape[1])
        key_indices = torch.arange(seq_len, device=valid_mask.device)
        allowed = (
            key_indices[None, None, :] <= action_indices[:, :, None]
        ) & valid_mask[:, None, :]
        return torch.zeros(
            (*allowed.shape[:2], seq_len),
            device=valid_mask.device,
            dtype=dtype,
        ).masked_fill(~allowed, torch.finfo(dtype).min)[:, None]

    def _layout(
        self,
        condition: Qwen35ActionCondition,
        action_len: int,
    ) -> dict[str, torch.Tensor | int]:
        mask = condition.prefix_attention_mask
        batch_size, prefix_width = map(int, mask.shape)
        lengths = mask.sum(dim=1, dtype=torch.long)
        proprio_count = 0 if condition.proprio_token is None else 1
        total_lengths = lengths + proprio_count + int(action_len)
        # prepare_qwen_inputs right-pads exactly to the longest row, so at
        # least one prefix has prefix_width valid tokens.  Deriving the packed
        # width from that static shape avoids a GPU->CPU max().item() sync.
        self._assert_scalar(
            lengths.max() == prefix_width,
            "Qwen prefix width must equal the longest valid prefix length.",
        )
        packed_len = prefix_width + proprio_count + int(action_len)
        device = mask.device

        valid = (
            torch.arange(packed_len, device=device).unsqueeze(0)
            < total_lengths.unsqueeze(1)
        )
        action_indices = (
            lengths[:, None]
            + proprio_count
            + torch.arange(int(action_len), device=device)[None]
        )
        proprio_indices = lengths if proprio_count else None

        packed_position_ids = torch.zeros(
            (3, batch_size, packed_len), device=device, dtype=torch.long
        )
        packed_position_ids[:, :, :prefix_width] = (
            condition.prefix_position_ids
            * mask.to(dtype=torch.long).unsqueeze(0)
        )
        base_position = condition.prefix_position_ids.masked_fill(
            ~mask.unsqueeze(0), -1
        ).amax(dim=(0, 2))
        continuation = base_position[:, None] + torch.arange(
            1,
            proprio_count + int(action_len) + 1,
            device=device,
            dtype=torch.long,
        )[None]
        continuation_indices = lengths[:, None] + torch.arange(
            proprio_count + int(action_len),
            device=device,
            dtype=torch.long,
        )[None]
        packed_position_ids.scatter_(
            2,
            continuation_indices.unsqueeze(0).expand(3, -1, -1),
            continuation.unsqueeze(0).expand(3, -1, -1),
        )

        return {
            "lengths": lengths,
            "total_lengths": total_lengths,
            "packed_len": packed_len,
            "valid_mask": valid,
            "action_indices": action_indices,
            "proprio_indices": proprio_indices,
            "position_ids": packed_position_ids,
        }

    def _pack_layer(
        self,
        *,
        prefix: torch.Tensor,
        action_hidden: torch.Tensor,
        condition: Qwen35ActionCondition,
        layout: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        batch_size = int(prefix.shape[0])
        packed_len = int(layout["packed_len"])
        lengths = layout["lengths"]
        action_indices = layout["action_indices"]
        if not isinstance(lengths, torch.Tensor) or not isinstance(action_indices, torch.Tensor):
            raise TypeError("Invalid Qwen action layout tensor fields.")
        prefix_width = int(prefix.shape[1])
        packed = F.pad(
            prefix * condition.prefix_attention_mask[:, :, None].to(prefix.dtype),
            (0, 0, 0, packed_len - prefix_width),
        )
        if condition.proprio_token is not None:
            proprio_indices = layout["proprio_indices"]
            if not isinstance(proprio_indices, torch.Tensor):
                raise TypeError("Missing Qwen action proprio indices.")
            packed[
                torch.arange(batch_size, device=packed.device), proprio_indices
            ] = condition.proprio_token[:, 0]
        packed.scatter_(
            1,
            action_indices[:, :, None].expand(-1, -1, self.hidden_dim),
            action_hidden,
        )
        return packed

    @staticmethod
    def _gather_action(
        packed: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return packed.gather(
            1,
            action_indices[:, :, None].expand(-1, -1, int(packed.shape[-1])),
        )

    def _run_block(
        self,
        block: nn.Module,
        packed: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        scalar_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        def block_forward(
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            mask: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            return block(
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
            )

        cos, sin = position_embeddings
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block_forward,
                packed,
                cos,
                sin,
                attention_mask,
                scalar_position_ids,
                use_reentrant=False,
            )
        return block_forward(
            packed,
            cos,
            sin,
            attention_mask,
            scalar_position_ids,
        )

    def _run_linear_mixer_action_suffix(
        self,
        mixer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full recurrent update but materialize output gates for action rows.

        Prefix K/V, decay and beta values all update GatedDeltaNet's recurrent
        state and are therefore retained. Prefix query outputs, z gates,
        gated RMSNorm and out-projection never reach the next action block, so
        their gradients are identically zero in the reference implementation.
        """

        if (
            attention_mask is not None
            and attention_mask.shape[1] > 1
            and attention_mask.shape[0] > 1
        ):
            hidden_states = (
                hidden_states * attention_mask[:, :, None]
            ).to(hidden_states.dtype)

        batch_size, seq_len, _ = hidden_states.shape
        action_len = int(action_indices.shape[1])
        if self.skip_discarded_prefix_query:
            # ``in_proj_qkv`` is ordered [Q,K,V].  DeltaNet state updates do
            # not depend on Q; only action query outputs survive the gather
            # below.  The causal depthwise convolution means those action Qs
            # additionally depend on the preceding kernel_size-1 projected
            # rows.  Project exactly that fixed-size window and retain full
            # K/V projections for the recurrent state.  Filling earlier Q
            # rows with zero changes only dead prefix outputs and leaves every
            # live output/gradient mathematically unchanged.
            conv_context = int(mixer.conv_kernel_size) - 1
            query_offsets = torch.arange(
                -conv_context,
                action_len,
                device=hidden_states.device,
                dtype=action_indices.dtype,
            )
            query_indices = action_indices[:, :1] + query_offsets[None]
            self._assert_scalar(
                (query_indices >= 0).all()
                & (query_indices < seq_len).all(),
                "Action suffix lacks the required causal-convolution context.",
            )
            query_input = hidden_states.gather(
                1,
                query_indices[:, :, None].expand(
                    -1, -1, int(hidden_states.shape[-1])
                ),
            )
            projection_weight = mixer.in_proj_qkv.weight
            projection_bias = mixer.in_proj_qkv.bias
            query_projection = F.linear(
                query_input,
                projection_weight[: mixer.key_dim],
                None
                if projection_bias is None
                else projection_bias[: mixer.key_dim],
            )
            query = hidden_states.new_zeros(
                (batch_size, seq_len, mixer.key_dim)
            ).scatter(
                1,
                query_indices[:, :, None].expand(
                    -1, -1, mixer.key_dim
                ),
                query_projection,
            )
            key_value = F.linear(
                hidden_states,
                projection_weight[mixer.key_dim :],
                None
                if projection_bias is None
                else projection_bias[mixer.key_dim :],
            )
            mixed_qkv = torch.cat((query, key_value), dim=-1).transpose(1, 2)
        else:
            mixed_qkv = mixer.in_proj_qkv(hidden_states).transpose(1, 2)
        if mixer.causal_conv1d_fn is not None:
            mixed_qkv = mixer.causal_conv1d_fn(
                x=mixed_qkv,
                weight=mixer.conv1d.weight.squeeze(1),
                bias=mixer.conv1d.bias,
                activation=mixer.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(mixer.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [mixer.key_dim, mixer.key_dim, mixer.value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch_size, seq_len, -1, mixer.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, -1, mixer.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, mixer.head_v_dim)

        beta_logits = mixer.in_proj_b(hidden_states)
        decay_logits = mixer.in_proj_a(hidden_states)
        beta = beta_logits.sigmoid()
        decay = -mixer.A_log.float().exp() * F.softplus(
            decay_logits.float() + mixer.dt_bias
        )
        if mixer.num_v_heads // mixer.num_k_heads > 1:
            repeats = mixer.num_v_heads // mixer.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        core_output, _ = mixer.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        core_action = core_output.gather(
            1,
            action_indices[:, :, None, None].expand(
                -1, -1, mixer.num_v_heads, mixer.head_v_dim
            ),
        )
        action_input = self._gather_action(hidden_states, action_indices)
        z_action = mixer.in_proj_z(action_input).reshape(
            batch_size, action_len, -1, mixer.head_v_dim
        )
        core_action = mixer.norm(
            core_action.reshape(-1, mixer.head_v_dim),
            z_action.reshape(-1, mixer.head_v_dim),
        ).reshape(batch_size, action_len, -1)
        return mixer.out_proj(core_action)

    @staticmethod
    def _rotate_single(
        states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = int(cos.shape[-1])
        rotary = states[..., :rotary_dim]
        passthrough = states[..., rotary_dim:]
        half = int(rotary.shape[-1]) // 2
        rotated_half = torch.cat(
            (-rotary[..., half:], rotary[..., :half]), dim=-1
        )
        rotated = rotary * cos + rotated_half * sin
        return torch.cat((rotated, passthrough), dim=-1)

    def _run_full_attention_action_suffix(
        self,
        mixer: nn.Module,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute full-attention K/V for all rows and Q/output for actions."""

        if float(mixer.attention_dropout) != 0.0:
            raise RuntimeError(
                "Action-suffix attention optimization requires zero attention dropout."
            )
        batch_size, seq_len, _ = hidden_states.shape
        action_len = int(action_indices.shape[1])
        action_input = self._gather_action(hidden_states, action_indices)

        action_shape = action_input.shape[:-1]
        query_states, gate = torch.chunk(
            mixer.q_proj(action_input).view(
                *action_shape, -1, mixer.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*action_shape, -1)
        query_states = mixer.q_norm(
            query_states.view(*action_shape, -1, mixer.head_dim)
        ).transpose(1, 2)
        key_states = mixer.k_norm(
            mixer.k_proj(hidden_states).view(
                batch_size, seq_len, -1, mixer.head_dim
            )
        ).transpose(1, 2)
        value_states = mixer.v_proj(hidden_states).view(
            batch_size, seq_len, -1, mixer.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        action_cos = cos.gather(
            1,
            action_indices[:, :, None].expand(-1, -1, cos.shape[-1]),
        )
        action_sin = sin.gather(
            1,
            action_indices[:, :, None].expand(-1, -1, sin.shape[-1]),
        )
        query_states = self._rotate_single(
            query_states, action_cos, action_sin
        )
        key_states = self._rotate_single(key_states, cos, sin)

        if mixer.num_key_value_groups > 1:
            key_states = key_states[:, :, None].expand(
                batch_size,
                key_states.shape[1],
                mixer.num_key_value_groups,
                seq_len,
                mixer.head_dim,
            ).reshape(
                batch_size, -1, seq_len, mixer.head_dim
            )
            value_states = value_states[:, :, None].expand(
                batch_size,
                value_states.shape[1],
                mixer.num_key_value_groups,
                seq_len,
                mixer.head_dim,
            ).reshape(
                batch_size, -1, seq_len, mixer.head_dim
            )

        if int(attention_mask.shape[-2]) == action_len:
            action_mask = attention_mask
        else:
            action_mask = attention_mask.gather(
                2,
                action_indices[:, None, :, None].expand(
                    -1, attention_mask.shape[1], -1, seq_len
                ),
            )
        implementation = str(
            getattr(mixer.config, "_attn_implementation", "eager")
        )
        if implementation == "sdpa":
            attention_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=action_mask,
                dropout_p=0.0,
                scale=mixer.scaling,
                is_causal=False,
            ).transpose(1, 2).contiguous()
        elif implementation == "eager":
            attention_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                * mixer.scaling
            )
            attention_weights = attention_weights + action_mask
            attention_weights = F.softmax(
                attention_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attention_output = torch.matmul(
                attention_weights, value_states
            ).transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                "Action-suffix attention optimization currently supports "
                f"eager/sdpa, got {implementation!r}."
            )
        attention_output = attention_output.reshape(
            batch_size, action_len, -1
        )
        attention_output = attention_output * torch.sigmoid(gate)
        return mixer.o_proj(attention_output)

    def _run_block_action_suffix(
        self,
        block: nn.Module,
        packed: torch.Tensor,
        *,
        action_indices: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        scalar_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return only live action rows while preserving the token mixer.

        The complete prefix must still enter self-attention/GatedDeltaNet so
        action tokens see exactly the same K/V or recurrent state.  After the
        token mixer, however, prefix rows never feed another action block: the
        next iteration repacks a fresh student-Qwen prefix.  Since Qwen's
        post-attention norm and MLP are row-wise, applying them only to the
        gathered action rows removes dead computation without changing the
        function or any non-zero parameter gradient.
        """

        def suffix_forward(
            hidden_states: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            mask: torch.Tensor,
            positions: torch.Tensor,
            suffix_indices: torch.Tensor,
        ) -> torch.Tensor:
            normalized = block.input_layernorm(hidden_states)
            if block.block_type == "linear_attention":
                mixed_action = self._run_linear_mixer_action_suffix(
                    block.linear_attn,
                    normalized,
                    attention_mask=mask,
                    action_indices=suffix_indices,
                )
            elif block.block_type == "full_attention":
                mixed_action = self._run_full_attention_action_suffix(
                    block.self_attn,
                    normalized,
                    position_embeddings=(cos, sin),
                    attention_mask=mask,
                    action_indices=suffix_indices,
                )
            else:
                raise RuntimeError(
                    f"Unsupported Qwen action block type: {block.block_type!r}."
                )

            action_residual = self._gather_action(hidden_states, suffix_indices)
            action_hidden = action_residual + mixed_action
            return action_hidden + block.mlp(
                block.post_attention_layernorm(action_hidden)
            )

        cos, sin = position_embeddings
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                suffix_forward,
                packed,
                cos,
                sin,
                attention_mask,
                scalar_position_ids,
                action_indices,
                use_reentrant=False,
            )
        return suffix_forward(
            packed,
            cos,
            sin,
            attention_mask,
            scalar_position_ids,
            action_indices,
        )

    def forward_native_mot(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        condition: Qwen35ActionCondition,
    ) -> torch.Tensor:
        action_hidden = self._embed_action(action_tokens, timestep)
        batch_size, action_len = map(int, action_hidden.shape[:2])
        if int(condition.prefix_attention_mask.shape[0]) != batch_size:
            raise ValueError(
                "Qwen action condition/action batch mismatch: "
                f"{condition.prefix_attention_mask.shape[0]} vs {batch_size}."
            )
        layout = self._layout(condition, action_len)
        valid_mask = layout["valid_mask"]
        action_indices = layout["action_indices"]
        position_ids = layout["position_ids"]
        packed_len = int(layout["packed_len"])
        if not all(
            isinstance(item, torch.Tensor)
            for item in (valid_mask, action_indices, position_ids)
        ):
            raise TypeError("Invalid Qwen action layout.")

        # Rotary values depend on dtype/device/positions, not hidden values.
        # Qwen rotary_emb uses x only for device/dtype; position_ids determine
        # the full sequence shape.  Reusing action_hidden avoids allocating a
        # discarded B x packed_len x hidden tensor (about 40 MiB at b48).
        position_embeddings = self.rotary_emb(action_hidden, position_ids)
        if self.action_attention_mode == QWEN_ACTION_ATTENTION_CAUSAL:
            if self.skip_discarded_prefix_mlp:
                full_attention_mask = self._build_action_suffix_causal_mask(
                    valid_mask,
                    action_indices,
                    dtype=action_hidden.dtype,
                )
            else:
                full_attention_mask = self._build_full_causal_mask(
                    valid_mask,
                    dtype=action_hidden.dtype,
                )
        else:  # guarded in __init__; retain a loud failure for loaded objects.
            raise RuntimeError(
                f"Unsupported Qwen action attention mode: {self.action_attention_mode!r}."
            )
        scalar_position_ids = position_ids[0]

        for layer_idx, (block, prefix, layer_type) in enumerate(
            zip(self.blocks, condition.layerwise_prefix_tokens, self.layer_types)
        ):
            packed = self._pack_layer(
                prefix=prefix,
                action_hidden=action_hidden,
                condition=condition,
                layout=layout,
            )
            if layer_type == "linear_attention":
                block_mask = valid_mask
            elif layer_type == "full_attention":
                block_mask = full_attention_mask
            else:
                raise RuntimeError(
                    f"Unsupported Qwen action layer type at {layer_idx}: {layer_type!r}."
                )
            if self.skip_discarded_prefix_mlp:
                action_hidden = self._run_block_action_suffix(
                    block,
                    packed,
                    action_indices=action_indices,
                    position_embeddings=position_embeddings,
                    attention_mask=block_mask,
                    scalar_position_ids=scalar_position_ids,
                )
            else:
                packed = self._run_block(
                    block,
                    packed,
                    position_embeddings=position_embeddings,
                    attention_mask=block_mask,
                    scalar_position_ids=scalar_position_ids,
                )
                action_hidden = self._gather_action(packed, action_indices)

        return self.head(self.final_norm(action_hidden))

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        condition: Qwen35ActionCondition,
    ) -> torch.Tensor:
        return self.forward_native_mot(
            action_tokens=action_tokens,
            timestep=timestep,
            condition=condition,
        )


__all__ = [
    "ActionDiTQwen35Native",
    "Qwen35ActionCondition",
    "QWEN_ACTION_ATTENTION_CAUSAL",
    "QWEN_ACTION_ATTENTION_MODES",
]
