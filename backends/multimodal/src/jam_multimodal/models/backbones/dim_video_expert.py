from __future__ import annotations

import os
from typing import Any, Dict

import pyrallis
import torch
import torch.nn as nn
from safetensors.torch import load_file

from jam_multimodal.utils.logging_config import get_logger

logger = get_logger(__name__)


class DimVideoExpert(nn.Module):
    """JAMPolicy wrapper around DIM's SANA image-editing backbone."""

    block_protocol = "sana"

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        config,
        max_condition_length: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model = model
        self.vae = vae
        self.config = config
        self.max_condition_length = int(max_condition_length)
        self.torch_dtype = torch_dtype
        self.blocks = model.blocks
        self.hidden_dim = int(model.hidden_size)
        self.caption_dim = int(config.text_encoder.caption_channels)
        self.context_dim = self.hidden_dim
        # DIM's released config uses LiteLA: heads = hidden_size // linear_head_dim.
        linear_head_dim = int(getattr(config.model, "linear_head_dim", 32))
        self.num_heads = self.hidden_dim // linear_head_dim
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = linear_head_dim
        self.use_gradient_checkpointing = bool(getattr(config.train, "grad_checkpointing", False))
        self.vae_scale_factor = float(getattr(config.vae, "scale_factor", 0.41407) or 0.41407)
        self.vae_downsample_rate = int(getattr(config.vae, "vae_downsample_rate", 32))
        self.vae_latent_dim = int(getattr(config.vae, "vae_latent_dim", 32))
        self._expand_context_if_needed(self.max_condition_length)

    @classmethod
    def from_pretrained(
        cls,
        dim_model_path: str,
        sana_config_path: str,
        max_condition_length: int = 8192,
        with_latents_condition: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DimVideoExpert":
        from models.diffusion.model.builder import build_model, get_vae
        from models.diffusion.model.nets import sana_multi_scale as _sana_multi_scale  # noqa: F401
        from models.diffusion.model.utils import get_weight_dtype
        from models.diffusion.utils.config import SanaConfig, model_init_config

        config = pyrallis.load(SanaConfig, open(sana_config_path))
        latent_size = int(config.model.image_size) // int(config.vae.vae_downsample_rate)
        model_kwargs = model_init_config(config, latent_size=latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=bool(config.model.get("fp32_attention", False)) and config.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        if with_latents_condition:
            cls._expand_model_channels(model)
        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        model = model.to(device=device, dtype=weight_dtype)
        vae_dtype = get_weight_dtype(config.vae.weight_dtype)
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device).to(vae_dtype)
        expert = cls(
            model=model,
            vae=vae,
            config=config,
            max_condition_length=max_condition_length,
            torch_dtype=torch_dtype,
        )
        expert.load_dim_checkpoint(dim_model_path)
        expert.to(device=device, dtype=torch_dtype)
        expert.vae.to(device=device, dtype=vae_dtype).eval()
        return expert

    @staticmethod
    def _expand_model_channels(model: nn.Module, zero_init: bool = False) -> None:
        conv_in = model.x_embedder.proj
        in_ch = int(conv_in.in_channels)
        out_ch = int(conv_in.out_channels)
        if in_ch % 2 == 0 and in_ch > 32:
            return
        new_conv = nn.Conv2d(
            in_channels=in_ch * 2,
            out_channels=out_ch,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
        )
        with torch.no_grad():
            new_conv.weight[:, :in_ch].copy_(conv_in.weight)
            if zero_init:
                new_conv.weight[:, in_ch:].zero_()
            else:
                new_conv.weight[:, in_ch:].copy_(conv_in.weight)
            new_conv.bias.copy_(conv_in.bias)
        model.x_embedder.proj = new_conv

    def _expand_context_if_needed(self, token_num: int) -> None:
        null_embedding = self.model.y_embedder.y_embedding
        if int(null_embedding.shape[0]) == int(token_num):
            return
        in_channels = int(null_embedding.shape[1])
        new_embedding = torch.randn(token_num, in_channels, device=null_embedding.device, dtype=null_embedding.dtype)
        new_embedding = new_embedding / (in_channels**0.5)
        with torch.no_grad():
            copy_len = min(int(null_embedding.shape[0]), int(token_num))
            new_embedding[:copy_len].copy_(null_embedding[:copy_len])
        self.model.y_embedder.y_embedding = nn.Parameter(new_embedding)

    def load_dim_checkpoint(self, dim_model_path: str) -> None:
        ckpt = os.path.join(dim_model_path, "model.safetensors") if os.path.isdir(dim_model_path) else dim_model_path
        state = load_file(ckpt)
        model_state = {k[len("decoder.model.") :]: v for k, v in state.items() if k.startswith("decoder.model.")}
        y_embedding_key = "y_embedder.y_embedding"
        if y_embedding_key in model_state:
            checkpoint_y_embedding = model_state[y_embedding_key]
            current_y_embedding = self.model.y_embedder.y_embedding
            if tuple(checkpoint_y_embedding.shape) != tuple(current_y_embedding.shape):
                if checkpoint_y_embedding.ndim != current_y_embedding.ndim or checkpoint_y_embedding.shape[1:] != current_y_embedding.shape[1:]:
                    raise RuntimeError(
                        "DIM checkpoint y_embedder.y_embedding has incompatible shape "
                        f"{tuple(checkpoint_y_embedding.shape)} for current shape {tuple(current_y_embedding.shape)}."
                    )
                adapted_y_embedding = current_y_embedding.detach().to(
                    device=checkpoint_y_embedding.device,
                    dtype=checkpoint_y_embedding.dtype,
                ).clone()
                copy_len = min(int(checkpoint_y_embedding.shape[0]), int(current_y_embedding.shape[0]))
                adapted_y_embedding[:copy_len].copy_(checkpoint_y_embedding[:copy_len])
                model_state[y_embedding_key] = adapted_y_embedding
                logger.info(
                    "Adapted DIM y_embedder.y_embedding from checkpoint shape %s to current shape %s.",
                    tuple(checkpoint_y_embedding.shape),
                    tuple(current_y_embedding.shape),
                )
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        if missing:
            logger.warning("DIM SANA missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("DIM SANA unexpected keys: %s", unexpected[:20])

        vae_state = {k[len("decoder.vae.") :]: v for k, v in state.items() if k.startswith("decoder.vae.")}
        if vae_state:
            missing_vae, unexpected_vae = self.vae.load_state_dict(vae_state, strict=False)
            if missing_vae:
                logger.info("DIM VAE missing keys: %s", missing_vae[:20])
            if unexpected_vae:
                logger.info("DIM VAE unexpected keys: %s", unexpected_vae[:20])

    def _pad_context(self, context: torch.Tensor, context_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}")
        max_len = self.max_condition_length
        if context.shape[1] > max_len:
            context = context[:, -max_len:]
            context_mask = context_mask[:, -max_len:]
        elif context.shape[1] < max_len:
            pad_len = max_len - context.shape[1]
            context = torch.cat([context, context.new_zeros(context.shape[0], pad_len, context.shape[2])], dim=1)
            context_mask = torch.cat(
                [context_mask, torch.zeros(context_mask.shape[0], pad_len, device=context_mask.device, dtype=torch.bool)],
                dim=1,
            )
        return context, context_mask.to(dtype=torch.bool)

    def prepare_context(self, context: torch.Tensor, context_mask: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
        context, context_mask = self._pad_context(context, context_mask)
        y = context.unsqueeze(1).to(dtype=self.torch_dtype)
        mask4 = context_mask[:, None, None, :]
        y = self.model.y_embedder(y, self.training, mask=mask4)
        if getattr(self.model, "y_norm", False):
            y = self.model.attention_y_norm(y)
        y_tokens = y.squeeze(1)
        try:
            from models.diffusion.model.nets.sana_blocks import _xformers_available
        except Exception:
            _xformers_available = False
        if _xformers_available:
            y_for_cross = y_tokens.masked_select(context_mask.unsqueeze(-1) != 0).view(1, -1, y_tokens.shape[-1])
            cross_mask = context_mask.sum(dim=1).tolist()
        else:
            y_for_cross = y
            cross_mask = context_mask
        return y_for_cross, cross_mask, y_tokens, context_mask

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        latents_condition: torch.Tensor | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if x.ndim != 4:
            raise ValueError(f"`x` must be [B,C,H,W], got {tuple(x.shape)}")
        if latents_condition is not None:
            if tuple(latents_condition.shape) != tuple(x.shape):
                raise ValueError(
                    f"`latents_condition` shape must match x, got {tuple(latents_condition.shape)} vs {tuple(x.shape)}"
                )
            x_in = torch.cat([x, latents_condition.to(dtype=x.dtype, device=x.device)], dim=1)
        else:
            x_in = x
        self.model.h, self.model.w = x.shape[-2] // self.model.patch_size, x.shape[-1] // self.model.patch_size
        tokens = self.model.x_embedder(x_in.to(dtype=self.torch_dtype))
        timestep = timestep.to(device=tokens.device)
        if getattr(self.model, "timestep_norm_scale_factor", 1.0) != 1.0:
            t = (timestep.float() / self.model.timestep_norm_scale_factor).to(torch.float32)
        else:
            t = timestep.long().to(torch.float32)
        t_emb = self.model.t_embedder(t)
        t_mod = self.model.t_block(t_emb)
        y_for_cross, cross_mask, y_tokens, y_mask = self.prepare_context(context, context_mask)
        return {
            "tokens": tokens,
            "t_emb": t_emb,
            "t_mod": t_mod,
            "context": y_for_cross,
            "context_mask": cross_mask,
            "context_tokens": y_tokens,
            "context_mask_bool": y_mask,
            "meta": {
                "batch_size": int(tokens.shape[0]),
                "height": int(x.shape[-2]),
                "width": int(x.shape[-1]),
                "token_height": int(self.model.h),
                "token_width": int(self.model.w),
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        out = self.model.final_layer(tokens, pre_state["t_emb"])
        return self.model.unpatchify(out)

    @torch.no_grad()
    def encode_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=next(self.vae.parameters()).device, dtype=getattr(self.vae, "dtype", image.dtype))
        z = self.vae.encode(image)[0]
        return (z * self.vae_scale_factor).to(dtype=self.torch_dtype)

    @torch.no_grad()
    def decode_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=next(self.vae.parameters()).device, dtype=getattr(self.vae, "dtype", latents.dtype))
        image = self.vae.decode(z / self.vae_scale_factor, return_dict=False)[0]
        return image.detach().float().clamp(-1, 1)
