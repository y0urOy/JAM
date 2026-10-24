from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from einops import rearrange

from jam_multimodal.utils.logging_config import get_logger

from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

logger = get_logger(__name__)


class OmniGen2VideoExpert(nn.Module):
    """JAMPolicy video expert wrapper around OmniGen2Transformer2DModel."""

    block_protocol = "omnigen2"

    def __init__(self, transformer: OmniGen2Transformer2DModel):
        super().__init__()
        self.transformer = transformer
        self.blocks = transformer.layers
        self.hidden_dim = int(transformer.config.hidden_size)
        self.num_heads = int(transformer.config.num_attention_heads)
        self.num_kv_heads = int(transformer.config.num_kv_heads)
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.use_gradient_checkpointing = bool(getattr(transformer, "gradient_checkpointing", False))

        self.freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            axes_dim=tuple(transformer.config.axes_dim_rope),
            axes_lens=tuple(transformer.config.axes_lens),
            theta=int(transformer.rope_embedder.theta),
        )

    @classmethod
    def from_pretrained(cls, model_path: str, subfolder: str | None = "transformer", **kwargs) -> "OmniGen2VideoExpert":
        transformer = OmniGen2Transformer2DModel.from_pretrained(model_path, subfolder=subfolder, **kwargs)
        return cls(transformer)

    def _as_image_list(self, hidden_states: torch.Tensor | List[torch.Tensor]) -> List[torch.Tensor]:
        if isinstance(hidden_states, torch.Tensor):
            if hidden_states.ndim != 4:
                raise ValueError(f"`hidden_states` tensor must be [B,C,H,W], got {tuple(hidden_states.shape)}")
            return [item for item in hidden_states]
        return hidden_states

    def _as_ref_image_list(
        self,
        ref_image_hidden_states: torch.Tensor | List[List[torch.Tensor]] | None,
        batch_size: int,
    ) -> List[List[torch.Tensor] | None] | None:
        if ref_image_hidden_states is None:
            return None
        if isinstance(ref_image_hidden_states, torch.Tensor):
            if ref_image_hidden_states.ndim != 4:
                raise ValueError(
                    f"`ref_image_hidden_states` tensor must be [B,C,H,W], got {tuple(ref_image_hidden_states.shape)}"
                )
            if ref_image_hidden_states.shape[0] != batch_size:
                raise ValueError("Batch mismatch between hidden_states and ref_image_hidden_states.")
            return [[item] for item in ref_image_hidden_states]
        return ref_image_hidden_states

    def _build_prefix_rotary(
        self,
        context_mask: torch.Tensor,
        l_effective_ref_img_len: list[list[int]],
        ref_img_sizes: list[list[tuple[int, int]] | None],
        device: torch.device,
    ):
        transformer = self.transformer
        batch_size = int(context_mask.shape[0])
        p = int(transformer.config.patch_size)
        encoder_seq_len = int(context_mask.shape[1])
        cap_lens = context_mask.sum(dim=1).tolist()
        seq_lengths = [int(cap_len + sum(ref_lens)) for cap_len, ref_lens in zip(cap_lens, l_effective_ref_img_len)]
        max_seq_len = max(seq_lengths)
        max_ref_len = max([sum(ref_lens) for ref_lens in l_effective_ref_img_len])

        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)
        for i, cap_len in enumerate(cap_lens):
            position_ids[i, :cap_len] = torch.arange(cap_len, dtype=torch.int32, device=device)[:, None].repeat(1, 3)
            pe_shift = cap_len
            pe_shift_len = cap_len
            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    height, width = ref_img_size
                    h_tokens, w_tokens = height // p, width // p
                    row_ids = torch.arange(h_tokens, dtype=torch.int32, device=device)[:, None].repeat(1, w_tokens).flatten()
                    col_ids = torch.arange(w_tokens, dtype=torch.int32, device=device)[None, :].repeat(h_tokens, 1).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = col_ids
                    pe_shift += max(h_tokens, w_tokens)
                    pe_shift_len += ref_img_len

        rotary_emb = transformer.rope_embedder._get_freqs_cis(self.freqs_cis, position_ids)
        context_rotary_emb = torch.zeros(
            batch_size, encoder_seq_len, rotary_emb.shape[-1], device=device, dtype=rotary_emb.dtype
        )
        ref_img_rotary_emb = torch.zeros(
            batch_size, max_ref_len, rotary_emb.shape[-1], device=device, dtype=rotary_emb.dtype
        )
        for i, cap_len in enumerate(cap_lens):
            ref_len = sum(l_effective_ref_img_len[i])
            context_rotary_emb[i, :cap_len] = rotary_emb[i, :cap_len]
            ref_img_rotary_emb[i, :ref_len] = rotary_emb[i, cap_len : cap_len + ref_len]
        return context_rotary_emb, ref_img_rotary_emb, rotary_emb, cap_lens, seq_lengths

    def _ref_patch_embed_and_refine(
        self,
        ref_image_hidden_states: torch.Tensor,
        ref_img_rotary_emb: torch.Tensor,
        l_effective_ref_img_len: list[list[int]],
        temb: torch.Tensor,
    ) -> torch.Tensor:
        transformer = self.transformer
        ref_image_hidden_states = transformer.ref_image_patch_embedder(ref_image_hidden_states)
        batch_size = int(ref_image_hidden_states.shape[0])
        max_ref_img_len = int(ref_image_hidden_states.shape[1])
        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len] += transformer.image_index_embedding[j]
                shift += ref_img_len

        flat_ref_lens = [length for ref_lens in l_effective_ref_img_len for length in ref_lens]
        num_ref_images = len(flat_ref_lens)
        max_single_ref_len = max(flat_ref_lens)
        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_single_ref_len, dtype=torch.bool)
        batch_ref_states = ref_image_hidden_states.new_zeros(num_ref_images, max_single_ref_len, self.hidden_dim)
        batch_ref_rope = ref_image_hidden_states.new_zeros(
            num_ref_images, max_single_ref_len, ref_img_rotary_emb.shape[-1], dtype=ref_img_rotary_emb.dtype
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)
        idx = 0
        for i, ref_lens in enumerate(l_effective_ref_img_len):
            shift = 0
            for ref_len in ref_lens:
                batch_ref_img_mask[idx, :ref_len] = True
                batch_ref_states[idx, :ref_len] = ref_image_hidden_states[i, shift : shift + ref_len]
                batch_ref_rope[idx, :ref_len] = ref_img_rotary_emb[i, shift : shift + ref_len]
                batch_temb[idx] = temb[i]
                shift += ref_len
                idx += 1
        for layer in transformer.ref_image_refiner:
            batch_ref_states = layer(batch_ref_states, batch_ref_img_mask, batch_ref_rope, batch_temb)

        idx = 0
        for i, ref_lens in enumerate(l_effective_ref_img_len):
            shift = 0
            for ref_len in ref_lens:
                ref_image_hidden_states[i, shift : shift + ref_len] = batch_ref_states[idx, :ref_len]
                shift += ref_len
                idx += 1
        return ref_image_hidden_states[:, :max_ref_img_len]

    def pre_dit(
        self,
        x: torch.Tensor | List[torch.Tensor] | None,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        ref_image_hidden_states: torch.Tensor | List[List[torch.Tensor]] | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        transformer = self.transformer
        prefix_only = x is None
        if prefix_only:
            if ref_image_hidden_states is None:
                raise ValueError("`ref_image_hidden_states` is required when `x=None`.")
            batch_size = (
                int(ref_image_hidden_states.shape[0])
                if isinstance(ref_image_hidden_states, torch.Tensor)
                else len(ref_image_hidden_states)
            )
            hidden_states = None
        else:
            hidden_states = self._as_image_list(x)
            batch_size = len(hidden_states)
        ref_image_hidden_states = self._as_ref_image_list(ref_image_hidden_states, batch_size)
        ref_device_tensor = ref_image_hidden_states[0][0] if ref_image_hidden_states is not None else hidden_states[0]
        device = ref_device_tensor.device
        dtype = ref_device_tensor.dtype

        temb, text_hidden_states = transformer.time_caption_embed(timestep, context, dtype)
        if prefix_only:
            (
                _unused_hidden,
                ref_image_hidden_states,
                _unused_img_mask,
                _ref_img_mask,
                l_effective_ref_img_len,
                _unused_img_len,
                ref_img_sizes,
                _unused_img_sizes,
            ) = transformer.flat_and_pad_to_seq(
                [ref_image_hidden_states[i][0] for i in range(batch_size)],
                ref_image_hidden_states,
            )
            (
                context_rotary_emb,
                ref_img_rotary_emb,
                rotary_emb,
                encoder_seq_lengths,
                seq_lengths,
            ) = self._build_prefix_rotary(context_mask, l_effective_ref_img_len, ref_img_sizes, device)
            for layer in transformer.context_refiner:
                text_hidden_states = layer(text_hidden_states, context_mask, context_rotary_emb)
            ref_hidden_states = self._ref_patch_embed_and_refine(
                ref_image_hidden_states,
                ref_img_rotary_emb,
                l_effective_ref_img_len,
                temb,
            )
            max_seq_len = max(seq_lengths)
            joint_hidden_states = ref_hidden_states.new_zeros(batch_size, max_seq_len, self.hidden_dim)
            for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
                ref_len = seq_len - encoder_seq_len
                joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
                joint_hidden_states[i, encoder_seq_len:seq_len] = ref_hidden_states[i, :ref_len]
            return {
                "tokens": joint_hidden_states,
                "freqs": rotary_emb,
                "t_mod": temb,
                "context": None,
                "context_mask": None,
                "encoder_seq_lengths": encoder_seq_lengths,
                "seq_lengths": seq_lengths,
                "l_effective_ref_img_len": l_effective_ref_img_len,
                "l_effective_img_len": [0 for _ in range(batch_size)],
                "img_sizes": [None for _ in range(batch_size)],
                "prefix_only": True,
                "meta": {
                    "batch_size": batch_size,
                    "max_video_seq_len": max_seq_len,
                    "seq_lengths": seq_lengths,
                    "encoder_seq_lengths": encoder_seq_lengths,
                },
            }

        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = transformer.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = transformer.rope_embedder(
            self.freqs_cis,
            context_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        for layer in transformer.context_refiner:
            text_hidden_states = layer(text_hidden_states, context_mask, context_rotary_emb)

        combined_img_hidden_states = transformer.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        max_seq_len = max(seq_lengths)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.hidden_dim)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, : seq_len - encoder_seq_len]

        return {
            "tokens": joint_hidden_states,
            "freqs": rotary_emb,
            "t_mod": temb,
            "context": None,
            "context_mask": None,
            "encoder_seq_lengths": encoder_seq_lengths,
            "seq_lengths": seq_lengths,
            "l_effective_ref_img_len": l_effective_ref_img_len,
            "l_effective_img_len": l_effective_img_len,
            "img_sizes": img_sizes,
            "meta": {
                "batch_size": batch_size,
                "max_video_seq_len": max_seq_len,
                "seq_lengths": seq_lengths,
                "encoder_seq_lengths": encoder_seq_lengths,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        transformer = self.transformer
        hidden_states = transformer.norm_out(tokens, pre_state["t_mod"])
        p = int(transformer.config.patch_size)
        outputs = []
        for i, (img_size, img_len, seq_len) in enumerate(
            zip(pre_state["img_sizes"], pre_state["l_effective_img_len"], pre_state["seq_lengths"])
        ):
            height, width = img_size
            outputs.append(
                rearrange(
                    hidden_states[i][seq_len - img_len : seq_len],
                    "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                    h=height // p,
                    w=width // p,
                    p1=p,
                    p2=p,
                )
            )
        return torch.stack(outputs, dim=0)
