from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable

ensure_ovis_u1_remote_code_importable()

from ovis_u1_hf.modeling_yak import YakTransformer, timestep_embedding  # noqa: E402


class OvisU1VideoExpert(nn.Module):
    """JAMPolicy wrapper around Ovis-U1's Yak MMDiT backbone."""

    block_protocol = "yak"

    def __init__(self, transformer: YakTransformer):
        super().__init__()
        self.transformer = transformer
        self.double_blocks = transformer.double_blocks
        self.single_blocks = transformer.single_blocks
        self.hidden_dim = int(transformer.hidden_size)
        self.num_heads = int(transformer.num_heads)
        self.num_kv_heads = int(transformer.num_heads)
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.double_layers = len(self.double_blocks)
        self.single_layers = len(self.single_blocks)
        self.use_gradient_checkpointing = bool(getattr(transformer, "gradient_checkpointing", False))
        # Optional, experiment-only input adapter.  The official Ovis path
        # continues to use transformer.img_in on packed VAE latents.
        self.external_ssl_input_dim: int | None = None
        self.external_ssl_img_in: nn.Module | None = None

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_visual_generator(cls, visual_generator: nn.Module) -> "OvisU1VideoExpert":
        if hasattr(visual_generator, "get_backbone"):
            transformer = visual_generator.get_backbone()
        else:
            transformer = visual_generator.backbone
        return cls(transformer)

    def enable_external_ssl_input(self, input_dim: int) -> None:
        """Add a dense SSL-token input adapter without altering official Ovis."""

        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be positive, got {input_dim}.")
        if self.external_ssl_img_in is not None:
            if self.external_ssl_input_dim != input_dim:
                raise ValueError(
                    "External SSL input adapter already configured with "
                    f"dim={self.external_ssl_input_dim}, requested={input_dim}."
                )
            return
        ref = next(self.transformer.parameters())
        self.external_ssl_input_dim = input_dim
        self.external_ssl_img_in = nn.Linear(input_dim, self.hidden_dim).to(
            device=ref.device,
            dtype=ref.dtype,
        )

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"`latents` must be [B,C,H,W], got {tuple(latents.shape)}")
        return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def build_img_ids(
        batch_size: int,
        token_height: int,
        token_width: int,
        *,
        time_value: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(token_height, token_width, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(token_height, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(token_width, device=device, dtype=dtype)[None, :]
        ids[..., 0] = time_value
        return repeat(ids, "h w c -> b (h w) c", b=batch_size)

    @staticmethod
    def build_action_ids(
        batch_size: int,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        return (x * mask_float).sum(dim=1) / denom

    def _txt_in_with_mask(self, context: torch.Tensor, text_mask: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        refiner = self.transformer.txt_in
        if text_mask is None:
            return refiner(context)
        if bool(text_mask.all().item()):
            return refiner(context)

        # Ovis-U1 remote code casts mask.float(), which promotes bf16 activations
        # to fp32 before bf16 Linear layers. Reproduce the refiner path with a
        # dtype-preserving mask and a masked pooled vector.
        c = refiner.c_embedder(self._masked_mean(context, text_mask))
        if getattr(refiner, "enable_cross_attn", False):
            single_channels = int(refiner.in_channels) // int(refiner.length)
            x, y = torch.split(
                context,
                [single_channels, single_channels * (int(refiner.length) - 1)],
                dim=-1,
            )
            x = refiner.input_embedder(x)
            y = refiner.kv_embedder(y)
        else:
            x = refiner.input_embedder(context)
            y = None

        refiner_mask = text_mask
        if getattr(refiner, "enable_cls_token", False):
            batch_size = int(x.shape[0])
            cls = refiner.cls_token.expand(batch_size, -1, -1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.ones(batch_size, 1, device=text_mask.device, dtype=torch.bool)
            refiner_mask = torch.cat([cls_mask, text_mask], dim=1)

        if getattr(refiner, "enable_cross_attn", False):
            x = refiner.fusion(x, y, c)
        x = refiner.individual_token_refiner(x, c, refiner_mask)

        if getattr(refiner, "enable_cls_token", False):
            x_global = x[:, 0]
            x = x[:, 1:]
        else:
            x_global = self._masked_mean(x, text_mask)
        return {"txt_fea": x, "txt_fea_avg": x_global}

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        ref_image_hidden_states: torch.Tensor | None = None,
        target_img_ids: torch.Tensor | None = None,
        ref_img_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if x.ndim != 3:
            raise ValueError(f"`x` must be packed Yak image tokens [B,N,16], got {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,4096], got {tuple(context.shape)}")
        batch_size, target_len = int(x.shape[0]), int(x.shape[1])
        cond_len = 0 if ref_image_hidden_states is None else int(ref_image_hidden_states.shape[1])
        device = x.device
        dtype = x.dtype
        if target_img_ids is None:
            raise ValueError("`target_img_ids` is required for OvisU1VideoExpert.pre_dit().")
        if ref_image_hidden_states is not None and ref_img_ids is None:
            raise ValueError("`ref_img_ids` is required when `ref_image_hidden_states` is provided.")

        img = x if ref_image_hidden_states is None else torch.cat([x, ref_image_hidden_states], dim=1)
        img_ids = target_img_ids if ref_img_ids is None else torch.cat([target_img_ids, ref_img_ids], dim=1)
        if self.external_ssl_img_in is None:
            img = self.transformer.img_in(img)
        else:
            if int(img.shape[-1]) != int(self.external_ssl_input_dim):
                raise ValueError(
                    "External SSL Yak input dim mismatch: "
                    f"expected {self.external_ssl_input_dim}, got {img.shape[-1]}."
                )
            img = self.external_ssl_img_in(img)

        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        vec = self.transformer.time_in(timestep_embedding(timestep, 256))
        text_mask = None
        if context_mask is not None:
            if context_mask.ndim != 2 or tuple(context_mask.shape) != tuple(context.shape[:2]):
                raise ValueError(
                    "`context_mask` must be [B,L] matching context, "
                    f"got {tuple(context_mask.shape)} for context {tuple(context.shape)}"
                )
            text_mask = context_mask.to(device=device, dtype=torch.bool)
        txt_dict = self._txt_in_with_mask(context, text_mask)
        txt = txt_dict["txt_fea"]
        vec = vec + self.transformer.vector_in(txt_dict["txt_fea_avg"])

        txt_ids = torch.zeros(batch_size, txt.shape[1], 3, device=device, dtype=img_ids.dtype)
        pe = self.transformer.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        txt_pe, img_pe = torch.split(pe, [txt.shape[1], img.shape[1]], dim=2)

        return {
            "tokens": {"txt": txt, "img": img},
            "freqs": {"txt": txt_pe, "img": img_pe, "full": pe},
            "t_mod": vec.to(dtype=dtype),
            "context": None,
            "context_mask": text_mask,
            "text_mask": text_mask,
            "target_len": target_len,
            "cond_len": cond_len,
            "txt_len": int(txt.shape[1]),
            "meta": {
                "batch_size": batch_size,
                "target_len": target_len,
                "cond_len": cond_len,
                "txt_len": int(txt.shape[1]),
            },
        }

    def post_dit(self, tokens: Dict[str, torch.Tensor], pre_state: Dict[str, Any]) -> torch.Tensor:
        img = tokens["img"] if isinstance(tokens, dict) else tokens
        out = self.transformer.final_layer(img, pre_state["t_mod"])
        target_len = int(pre_state["target_len"])
        return out[:, :target_len]

    def native_forward_from_pre_state(self, pre_state: Dict[str, Any]) -> torch.Tensor:
        img = pre_state["tokens"]["img"]
        txt = pre_state["tokens"]["txt"]
        pe = pre_state["freqs"]["full"]
        vec = pre_state["t_mod"]
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
        stream = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            stream = block(stream, vec=vec, pe=pe)
        img = stream[:, txt.shape[1] :]
        return self.post_dit({"img": img, "txt": txt}, pre_state)
