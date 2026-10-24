import hashlib
import os
import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from jam_multimodal.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .foresight_yak import cosine_align_loss
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class JAMPolicy(torch.nn.Module):
    """Visual-backbone/action-expert policy for JAM and its generative controls."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        stack: str = "wan22",
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        trim_cached_text_padding: bool = False,
        pack_proprio_after_text: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.stack = str(stack)
        self.omnigen2_online_text_cache_compatible = bool(omnigen2_online_text_cache_compatible)
        self.qwen_context_len = int(qwen_context_len)
        # Opt-in because this shared class also serves non-FLUX stacks.  For a
        # right-padded cached FLUX context, removing columns that are masked in
        # every sample is mathematically equivalent to retaining them: no
        # valid query can attend those keys, and valid token positions stay
        # unchanged.  Perform the trim on CPU before the H2D copy.
        self.trim_cached_text_padding = bool(trim_cached_text_padding)
        self.pack_proprio_after_text = bool(pack_proprio_after_text)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for JAMPolicy.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for JAMPolicy.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="wan22",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @classmethod
    def from_omnigen2_pretrained(
        cls,
        omnigen2_model_path: str,
        omnigen2_vae_path: str,
        qwen_path: str | None,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        transformer_subfolder: str | None = "transformer",
        vae_subfolder: str | None = "vae",
        load_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
    ):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from transformers import AutoTokenizer, Qwen2_5_VLModel

        from .action_dit_omnigen2 import ActionDiTOmnigen2
        from .omnigen2_video_expert import OmniGen2VideoExpert

        video_expert = OmniGen2VideoExpert.from_pretrained(
            omnigen2_model_path,
            subfolder=transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_layers": len(video_expert.blocks),
            "num_heads": int(video_expert.num_heads),
            "num_kv_heads": int(video_expert.num_kv_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match OmniGen2 video expert value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTOmnigen2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        vae = AutoencoderKL.from_pretrained(
            omnigen2_vae_path,
            subfolder=vae_subfolder,
            torch_dtype=torch_dtype,
        )
        tokenizer = None
        text_encoder = None
        if load_text_encoder:
            logger.warning(
                "JAMPolicy-OmniGen2 is loading Qwen from `%s`. This follows OmniGen2 official training configs "
                "using upstream Qwen2.5-VL-3B-Instruct. It is not yet confirmed whether the official "
                "inference `mllm/` subfolder should be used instead for this JAMPolicy path.",
                qwen_path,
            )
            if qwen_path is None:
                raise ValueError("`qwen_path` is required when `load_text_encoder=True`.")
            tokenizer = AutoTokenizer.from_pretrained(qwen_path)
            tokenizer.padding_side = "right"
            text_encoder = Qwen2_5_VLModel.from_pretrained(qwen_path, torch_dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=2048,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="omnigen2",
            omnigen2_online_text_cache_compatible=bool(omnigen2_online_text_cache_compatible),
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "omnigen2_transformer": omnigen2_model_path,
            "omnigen2_vae": omnigen2_vae_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_ovis_u1_pretrained(
        cls,
        ovis_u1_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        load_condition_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
    ):
        from transformers import AutoModelForCausalLM

        from .action_dit_yak import ActionDiTYak
        from .mot import MoT
        from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable
        from .ovis_u1_video_expert import OvisU1VideoExpert

        ensure_ovis_u1_remote_code_importable()
        ovis_model = AutoModelForCausalLM.from_pretrained(
            ovis_u1_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).eval()
        if load_condition_encoder:
            ovis_model = ovis_model.to(device=device, dtype=torch_dtype)
        visual_generator = ovis_model.get_visual_generator()
        video_expert = OvisU1VideoExpert.from_visual_generator(visual_generator)

        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "residual_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match Ovis-U1 Yak value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTYak.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=visual_generator.get_vae(),
            text_encoder=None,
            tokenizer=None,
            text_dim=4096,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="ovis_u1",
        )
        object.__setattr__(model, "ovis_condition_encoder", ovis_model if load_condition_encoder else None)
        model.model_paths = {
            "ovis_u1": ovis_u1_model_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    @classmethod
    def from_flux2_klein_pretrained(
        cls,
        flux2_model_path: str,
        ae_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        proprio_dim: Optional[int] = None,
        load_text_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
        flux2_lora_config: Optional[dict[str, Any]] = None,
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 512,
        trim_cached_text_padding: bool = False,
    ):
        from safetensors.torch import load_file as load_sft

        from .action_dit_flux2 import ActionDiTFlux2
        from .flux2_imports import ensure_flux2_importable
        from .flux2_video_expert import Flux2VideoExpert
        from .mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = 7680
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = 12288
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        video_expert = Flux2VideoExpert.from_pretrained(
            flux2_model_path=flux2_model_path,
            variant=key,
            flux2_src_path=flux2_src_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        flux2_lora_config = dict(flux2_lora_config or {})
        if bool(flux2_lora_config.get("enabled", False)):
            from .lora import apply_lora_to_linear_suffixes

            target_suffixes = flux2_lora_config.get(
                "target_suffixes",
                [
                    "qkv",
                    "proj",
                    "linear1",
                    "linear2",
                    "img_mlp.0",
                    "img_mlp.2",
                    "txt_mlp.0",
                    "txt_mlp.2",
                ],
            )
            apply_lora_to_linear_suffixes(
                video_expert.transformer,
                target_suffixes=target_suffixes,
                rank=int(flux2_lora_config.get("rank", 16)),
                alpha=float(flux2_lora_config.get("alpha", 16.0)),
                dropout=float(flux2_lora_config.get("dropout", 0.0)),
            )
            video_expert.flux2_lora_enabled = True
            video_expert.flux2_lora_target_suffixes = tuple(str(item) for item in target_suffixes)
        else:
            video_expert.flux2_lora_enabled = False
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_heads": int(video_expert.num_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        action_cfg.setdefault("hidden_dim", 1024)
        for key_name, expected_value in expected_action_shape.items():
            if key_name in action_cfg and int(action_cfg[key_name]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match FLUX.2 value %s.",
                    key_name,
                    action_cfg[key_name],
                    expected_value,
                )
            action_cfg[key_name] = expected_value
        action_expert = ActionDiTFlux2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=True, assign=True)
        ae = ae.to(device=device, dtype=torch_dtype).eval()
        if load_text_encoder:
            from types import SimpleNamespace
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_spec = qwen3_model_spec or default_qwen3_model_spec
            qwen3_model = AutoModelForCausalLM.from_pretrained(
                model_spec,
                torch_dtype=torch_dtype,
            ).to(device).eval()
            text_encoder = SimpleNamespace(
                model=qwen3_model,
                tokenizer=AutoTokenizer.from_pretrained(model_spec),
                max_length=int(qwen_context_len),
            )
        else:
            text_encoder = None

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=ae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            trim_cached_text_padding=bool(trim_cached_text_padding),
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.model_paths = {
            "flux2": flux2_model_path,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "action_dit": action_dit_pretrained_path,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        model.save_lora_merged = bool(flux2_lora_config.get("save_lora_merged", bool(flux2_lora_config.get("enabled", False))))
        model.save_trainable_only = bool(flux2_lora_config.get("save_trainable_only", False))
        return model

    @classmethod
    def from_dim_pretrained(
        cls,
        dim_model_path: str,
        sana_config_path: str,
        qwen_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        proprio_dim: Optional[int] = None,
        max_condition_length: int = 8192,
        with_latents_condition: bool = True,
        load_mllm: bool = True,
        qwen_attn_implementation: str = "flash_attention_2",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
    ):
        from safetensors.torch import load_file
        from transformers import AutoProcessor

        from .action_dit_sana import ActionDiTSana
        from .dim_video_expert import DimVideoExpert
        from .mot import MoT

        from models.modeling_dim import Qwen2_5_VLForConditionalGeneration
        from models.multimodal.multimodal_projector.builder import build_projector

        video_expert = DimVideoExpert.from_pretrained(
            dim_model_path=dim_model_path,
            sana_config_path=sana_config_path,
            max_condition_length=int(max_condition_length),
            with_latents_condition=bool(with_latents_condition),
            device=device,
            torch_dtype=torch_dtype,
        )
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "attn_dim": int(video_expert.hidden_dim),
            "context_dim": int(video_expert.hidden_dim),
            "num_heads": int(video_expert.num_heads),
            "num_layers": len(video_expert.blocks),
        }
        for key, expected_value in expected_action_shape.items():
            if key in action_cfg and int(action_cfg[key]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match DIM/SANA value %s.",
                    key,
                    action_cfg[key],
                    expected_value,
                )
            action_cfg[key] = expected_value
        action_expert = ActionDiTSana.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        processor = AutoProcessor.from_pretrained(qwen_path, padding_side="left")
        processor.tokenizer.padding_side = "left"
        mllm = None
        if load_mllm:
            mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_path,
                torch_dtype=torch_dtype,
                attn_implementation=qwen_attn_implementation,
            ).eval()
            for param in mllm.parameters():
                param.requires_grad = False
        projector = build_projector(proj_in=2048, proj_out=int(video_expert.caption_dim), projector_type="mlp2x_gelu")
        ckpt = os.path.join(dim_model_path, "model.safetensors") if os.path.isdir(dim_model_path) else dim_model_path
        state = load_file(ckpt)
        projector_state = {k[len("projector.") :]: v for k, v in state.items() if k.startswith("projector.")}
        missing, unexpected = projector.load_state_dict(projector_state, strict=False)
        if missing:
            logger.warning("DIM projector missing keys: %s", missing[:20])
        if unexpected:
            logger.warning("DIM projector unexpected keys: %s", unexpected[:20])
        projector = projector.to(device=device, dtype=torch_dtype)

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=video_expert.vae,
            text_encoder=mllm,
            tokenizer=processor.tokenizer,
            text_dim=int(video_expert.caption_dim),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="dim",
            pack_proprio_after_text=bool(pack_proprio_after_text),
        )
        model.dim_processor = processor
        model.dim_projector = projector
        model.dim_max_condition_length = int(max_condition_length)
        model.dim_with_latents_condition = bool(with_latents_condition)
        model.model_paths = {
            "dim": dim_model_path,
            "sana_config": sana_config_path,
            "qwen": qwen_path,
            "action_dit": action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            if hasattr(self.text_encoder, "to"):
                self.text_encoder.to(*args, **kwargs)
            elif hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "to"):
                self.text_encoder.model.to(*args, **kwargs)
        if hasattr(self, "dim_projector"):
            self.dim_projector.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """Convert JAMPolicy scheduler timesteps back to Yak's [0, 1] time domain."""
        num_train_timesteps = float(getattr(scheduler, "num_train_timesteps", 1000))
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}.")
        return timestep / num_train_timesteps

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )

        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])

        # OmniGen2 treats context_mask.sum() as a contiguous prefix length, so the
        # proprio token must sit immediately after the valid text tokens.
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask,
            valid_rank,
            valid_counts[:, None] + 1 + invalid_rank,
        )
        new_context.scatter_(
            dim=1,
            index=target_indices[:, :, None].expand(-1, -1, context.shape[2]),
            src=context,
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]

        return new_context, new_context_mask

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size "
                f"({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )

    def _configure_action_attention_capture(
        self,
        *,
        condition_slice: slice | tuple[int, int] | list[int],
        condition_grid: tuple[int, int],
        prefix_len: int,
        action_len: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is None or not hasattr(capture, "configure"):
            return
        capture.configure(
            condition_slice=condition_slice,
            condition_grid=condition_grid,
            prefix_len=prefix_len,
            action_len=action_len,
            metadata=metadata or {},
        )

    def _start_action_attention_capture_step(self, step_idx: int) -> None:
        capture = getattr(self.mot, "action_attention_capture", None)
        if capture is not None and hasattr(capture, "start_step"):
            capture.start_step(int(step_idx))

    @staticmethod
    def _image_tensor_to_pil_list(image: torch.Tensor) -> list[Image.Image]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W], got {tuple(image.shape)}")
        image = image.detach().float().cpu()
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)
        arr = (image.permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [Image.fromarray(item) for item in arr]

    def _build_dim_user_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        tokenizer = self.dim_processor.tokenizer
        im_start_id = tokenizer("<|im_start|>").input_ids[0]
        im_end_id = tokenizer("<|im_end|>").input_ids[0]
        user_id = tokenizer("user").input_ids[0]
        newline_id = tokenizer("\n").input_ids[0]
        labels = torch.full_like(input_ids, -100, dtype=torch.long)
        for b in range(input_ids.shape[0]):
            user_begin, user_end = [], []
            for idx in range(3, input_ids.shape[1]):
                if (
                    input_ids[b, idx - 3] == im_start_id
                    and input_ids[b, idx - 2] == user_id
                    and input_ids[b, idx - 1] == newline_id
                ):
                    user_begin.append(idx)
                if input_ids[b, idx] == im_end_id and len(user_begin) == len(user_end) + 1:
                    user_end.append(idx)
            if len(user_begin) != len(user_end):
                raise ValueError("DIM Qwen user span parse failed: begin/end counts mismatch.")
            for start, end in zip(user_begin, user_end):
                labels[b, start:end] = input_ids[b, start:end]
        return labels

    def _build_dim_messages(self, instruction: str | Sequence[str], image: torch.Tensor):
        pil_images = self._image_tensor_to_pil_list(image)
        if isinstance(instruction, str):
            instructions = [instruction] * len(pil_images)
        else:
            instructions = list(instruction)
        if len(instructions) != len(pil_images):
            raise ValueError(f"Expected {len(pil_images)} DIM instructions, got {len(instructions)}.")
        messages = []
        for img, prompt in zip(pil_images, instructions):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": str(prompt)},
                        ],
                    }
                ]
            )
        return messages

    def _prepare_dim_condition_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or not hasattr(self, "dim_projector"):
            raise ValueError("DIM online conditioning requires loaded Qwen MLLM and projector.")
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.text_encoder(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
        projected = self.dim_projector(hidden.to(dtype=self.torch_dtype))
        user_labels = self._build_dim_user_labels(batch["input_ids"]).to(projected.device)
        valid = (user_labels != -100) & batch["attention_mask"].to(device=projected.device, dtype=torch.bool)
        max_len = int(getattr(self, "dim_max_condition_length", projected.shape[1]))
        out = projected.new_zeros(projected.shape[0], max_len, projected.shape[-1])
        mask = torch.zeros(projected.shape[0], max_len, device=projected.device, dtype=torch.bool)
        for b in range(projected.shape[0]):
            item = projected[b, valid[b]]
            if item.shape[0] > max_len:
                item = item[-max_len:]
            out[b, : item.shape[0]] = item
            mask[b, : item.shape[0]] = True
        return out, mask

    def _encode_dim_condition_online(
        self,
        instruction: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self._build_dim_messages(instruction, image)
        texts = [
            self.dim_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        pil_images = [msg[0]["content"][0]["image"] for msg in messages]
        batch = self.dim_processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        return self._prepare_dim_condition_from_batch(batch)

    @torch.no_grad()
    def _encode_dim_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        return self.video_expert.encode_image_latents(image.to(device=self.device, dtype=self.torch_dtype))

    @torch.no_grad()
    def _decode_dim_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.video_expert.decode_image_latents(latents)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "JAMPolicy training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for JAMPolicy training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        if action_dim_is_pad is not None:
            if action_dim_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_dim_is_pad']` must be 2D [B, D], got shape {tuple(action_dim_is_pad.shape)}"
                )
            if action_dim_is_pad.shape[0] != batch_size or action_dim_is_pad.shape[1] != action.shape[2]:
                raise ValueError(
                    "`sample['action_dim_is_pad']` shape mismatch: "
                    f"got {tuple(action_dim_is_pad.shape)} vs expected ({batch_size}, {action.shape[2]})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _encode_omnigen2_image_latents(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        return latent.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def _decode_omnigen2_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        decoded = self.vae.decode(z)
        image = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_omnigen2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        if "text_hidden_states" in sample and "text_attention_mask" in sample:
            return (
                sample["text_hidden_states"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                sample["text_attention_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        if "text_ids" in sample and "text_mask" in sample:
            text_ids = sample["text_ids"].to(device=self.device, non_blocking=True)
            text_mask = sample["text_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if self.text_encoder is None or self.tokenizer is None:
                raise ValueError(
                    "OmniGen2 stack needs precomputed text_hidden_states/text_attention_mask "
                    "or a loaded Qwen tokenizer/text_encoder."
                )
            instruction = sample.get("instruction")
            if instruction is None:
                raise ValueError("OmniGen2 stack sample must include `instruction` when text is not precomputed.")
            if self.omnigen2_online_text_cache_compatible:
                encoded = self.tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=self.qwen_context_len,
                    return_tensors="pt",
                )
            else:
                encoded = self.tokenizer(
                    instruction,
                    padding=True,
                    return_tensors="pt",
                )
            text_ids = encoded.input_ids.to(self.device)
            text_mask = encoded.attention_mask.to(self.device, dtype=torch.bool)
        text_hidden_states = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask,
            output_hidden_states=False,
        ).last_hidden_state
        return text_hidden_states.to(dtype=self.torch_dtype), text_mask

    @torch.no_grad()
    def _prepare_omnigen2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_omnigen2_text({"instruction": [prompt]})
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_omnigen2(self, sample, tiled: bool = False):
        debug_forward = (
            int(os.environ.get("JAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0") > 0
            or float(os.environ.get("JAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0") > 0
        )
        build_profile: list[tuple[str, float]] = []

        def _sync_and_mark(name: str, start_time: float) -> float:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            build_profile.append((name, now - start_time))
            return now

        t0 = time.perf_counter()
        video = sample.get("video")
        if "target_latent" in sample:
            target_latent = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("OmniGen2 stack sample requires `next_frame`, `target_image`, `video`, or `target_latent`.")
            target_latent = self._encode_omnigen2_image_latents(next_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.target_latent", t0)

        if "ref_image_latents" in sample:
            ref_image_latents = sample["ref_image_latents"]
            if isinstance(ref_image_latents, torch.Tensor):
                ref_image_latents = ref_image_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("OmniGen2 stack sample requires `current_frame`, `input_image`, `video`, or `ref_image_latents`.")
            ref_image_latents = self._encode_omnigen2_image_latents(current_frame)
        if debug_forward:
            t0 = _sync_and_mark("build.ref_latent", t0)

        text_hidden_states, text_attention_mask = self._encode_omnigen2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="OmniGen2 training sample",
            )
        if debug_forward:
            t0 = _sync_and_mark("build.text_to_device", t0)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if debug_forward:
            _sync_and_mark("build.action_to_device", t0)
            self._last_omnigen2_build_profile = build_profile
        return {
            "target_latent": target_latent,
            "ref_image_latents": ref_image_latents,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _encode_ovis_u1_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        image = image.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype), non_blocking=True)
        latent = self.vae.encode(image).latent_dist.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latent = latent - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latent = latent * self.vae.config.scaling_factor
        latent = latent.to(dtype=self.torch_dtype)
        tokens = OvisU1VideoExpert.pack_latents(latent)
        _, _, latent_h, latent_w = latent.shape
        ids = OvisU1VideoExpert.build_img_ids(
            batch_size=int(latent.shape[0]),
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=float(time_value),
            device=latent.device,
            dtype=latent.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_ovis_u1_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        latents = rearrange(
            tokens,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=latent_h // 2,
            w=latent_w // 2,
            ph=2,
            pw=2,
        )
        z = latents.to(device=self.device, dtype=getattr(self.vae, "dtype", self.torch_dtype))
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z = z / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z = z + self.vae.config.shift_factor
        image = self.vae.decode(z, return_dict=False)[0]
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _prepare_ovis_u1_vlm_image_tensors(
        self,
        image: torch.Tensor,
        visual_tokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        image = image.to(device=visual_tokenizer.device, dtype=torch.float32, non_blocking=True)
        if image.min() < 0:
            image = (image + 1.0) * 0.5
        image = image.clamp(0, 1)

        patch_size = int(visual_tokenizer.image_processor.patch_size)
        temporal_patch_size = int(visual_tokenizer.image_processor.temporal_patch_size)
        if temporal_patch_size != 1:
            raise ValueError("Ovis-U1 tensor online condition path currently expects temporal_patch_size=1.")
        hidden_stride = int(visual_tokenizer.image_processor.hidden_stride)
        min_pixels = int(visual_tokenizer.image_processor.min_pixels)
        max_pixels = int(visual_tokenizer.image_processor.max_pixels)
        _, _, height, width = image.shape
        resized_height, resized_width = visual_tokenizer.smart_resize(
            int(height),
            int(width),
            factor=patch_size * hidden_stride,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = F.interpolate(
            image,
            size=(int(resized_height), int(resized_width)),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)

        mean = torch.tensor(
            visual_tokenizer.image_processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            visual_tokenizer.image_processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        image = (image - mean) / std

        batch_size, channel, resized_height, resized_width = image.shape
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        if grid_h % hidden_stride != 0 or grid_w % hidden_stride != 0:
            raise ValueError(
                "Ovis-U1 VLM resized grid must be divisible by hidden_stride, "
                f"got grid=({grid_h},{grid_w}) hidden_stride={hidden_stride}."
            )
        patches = image.reshape(
            batch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
        flatten_patches = patches.reshape(batch_size * grid_h * grid_w, channel * patch_size * patch_size)
        grid_thws = torch.tensor(
            [[1, grid_h, grid_w]],
            device=image.device,
            dtype=torch.long,
        ).repeat(batch_size, 1)
        return flatten_patches, grid_thws, int(resized_height), int(resized_width)

    @torch.no_grad()
    def _encode_ovis_u1_condition_online(
        self,
        prompt: str | Sequence[str],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_encoder = getattr(self, "ovis_condition_encoder", None)
        if condition_encoder is None:
            raise ValueError(
                "Ovis-U1 online condition encoding requires `load_condition_encoder=true`. "
                "Otherwise provide precomputed `text_hidden_states`/`text_attention_mask`."
            )
        if image.ndim == 3:
            batch_size = 1
        elif image.ndim == 4:
            batch_size = int(image.shape[0])
        else:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, Sequence):
            prompts = list(prompt)
            if len(prompts) != batch_size:
                raise ValueError(f"Expected {batch_size} prompts, got {len(prompts)}.")
        else:
            raise ValueError(f"`prompt` must be str or sequence of str, got {type(prompt)}")
        text_tokenizer = condition_encoder.get_text_tokenizer()
        visual_tokenizer = condition_encoder.get_visual_tokenizer()
        pixel_values, grid_thws, _, _ = self._prepare_ovis_u1_vlm_image_tensors(
            image,
            visual_tokenizer,
        )
        prefix_ids = text_tokenizer(
            "<|im_start|>user\n",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device=self.device)
        suffix_texts = ["\n" + str(item).strip() + "<|im_end|>\n" for item in prompts]
        suffix = text_tokenizer(
            suffix_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(condition_encoder, "device", next(condition_encoder.parameters()).device)
        prefix_ids = prefix_ids.to(device=model_device).expand(batch_size, -1)
        image_placeholders = torch.tensor(
            visual_tokenizer.construct_image_placeholders((1, 1)),
            device=model_device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        suffix_ids = suffix.input_ids.to(device=model_device)
        suffix_mask = suffix.attention_mask.to(device=model_device, dtype=torch.bool)
        input_ids = torch.cat([prefix_ids, image_placeholders, suffix_ids], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones_like(prefix_ids, dtype=torch.bool),
                torch.ones_like(image_placeholders, dtype=torch.bool),
                suffix_mask,
            ],
            dim=1,
        )
        pixel_values = pixel_values.to(
            device=visual_tokenizer.device,
            dtype=getattr(visual_tokenizer, "dtype", self.torch_dtype),
        )
        grid_thws = grid_thws.to(device=visual_tokenizer.device)
        _, inputs_embeds, labels, merged_attention_mask, _ = condition_encoder.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=None,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            left_padding=True,
        )
        inputs_embeds = inputs_embeds.detach()
        if os.environ.get("JAM_OVIS_EMPTY_CACHE_EVERY_BATCH", "1") == "1":
            # Historical Ovis-U1 path.  This forces a device synchronization and
            # allocator purge every batch; keep it available for memory fallback,
            # but benchmark disabling it on GPUs with ample memory.
            torch.cuda.empty_cache()
        llm = condition_encoder.get_llm()
        llm_device = llm.device
        llm_inputs = {
            "inputs_embeds": inputs_embeds.to(llm_device),
            "attention_mask": merged_attention_mask.to(llm_device),
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        if os.environ.get("JAM_OVIS_CONDITION_BACKBONE_ONLY", "0") == "1":
            # We only consume hidden states.  Qwen3ForCausalLM's LM head and
            # all-IGNORE label loss do not contribute to those states and cost a
            # large [B,L,vocab] projection.  Calling its exact decoder backbone
            # preserves the hidden-state computation while skipping dead logits.
            llm_backbone = getattr(llm, "model", None)
            if llm_backbone is None:
                raise RuntimeError(
                    "Ovis condition backbone-only mode requires the CausalLM to expose `.model`."
                )
            outputs = llm_backbone(**llm_inputs)
        else:
            outputs = llm(
                **llm_inputs,
                labels=labels.to(llm_device),
            )
        text_hidden_states = torch.cat([outputs.hidden_states[-1], outputs.hidden_states[-2]], dim=-1).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        text_attention_mask = merged_attention_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return text_hidden_states, text_attention_mask

    @torch.no_grad()
    def _read_external_ssl_qwen_text_cache(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one frozen FLUX.2 Qwen3-4B instruction context.

        This path is deliberately limited to the opt-in DINO-native language
        ablation.  Qwen sees the instruction only; current images continue to
        enter exclusively through the trainable DINOv2 visual stream.
        """

        source = str(
            getattr(self, "external_ssl_text_condition_source", "ovis_multimodal")
        )
        if source != "qwen3_4b_flux2_cached":
            raise RuntimeError(
                "External SSL Qwen cache requested for incompatible text "
                f"source {source!r}."
            )
        cache_dir = getattr(self, "external_ssl_qwen_text_cache_dir", None)
        context_len = int(
            getattr(self, "external_ssl_qwen_context_len", 0)
        )
        hidden_dim = int(
            getattr(self, "external_ssl_qwen_hidden_dim", 0)
        )
        if not cache_dir or context_len <= 0 or hidden_dim <= 0:
            raise RuntimeError(
                "External SSL Qwen cache metadata is incomplete: "
                f"dir={cache_dir!r}, shape=({context_len},{hidden_dim})."
            )

        prompt = str(prompt)
        cache = getattr(self, "_external_ssl_text_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            object.__setattr__(self, "_external_ssl_text_cache", cache)
        pair = cache.get(prompt)
        if pair is None:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = os.path.join(
                str(cache_dir),
                f"{digest}.qwen3_flux2_len{context_len}.pt",
            )
            if not os.path.isfile(cache_path):
                raise FileNotFoundError(
                    "Missing DINO-native Qwen3 text cache: "
                    f"{cache_path}. Run the FLUX.2 Qwen3 cache precompute first."
                )
            payload = torch.load(cache_path, map_location="cpu")
            context = payload["text_hidden_states"]
            mask = payload["text_attention_mask"].to(dtype=torch.bool)
            if tuple(context.shape) != (context_len, hidden_dim):
                raise ValueError(
                    f"Cached Qwen3 context shape mismatch in {cache_path}: "
                    f"{tuple(context.shape)} vs {(context_len, hidden_dim)}."
                )
            if tuple(mask.shape) != (context_len,):
                raise ValueError(
                    f"Cached Qwen3 mask shape mismatch in {cache_path}: "
                    f"{tuple(mask.shape)} vs {(context_len,)}."
                )
            if not bool(mask.any()):
                raise ValueError(f"Cached Qwen3 context has no valid token: {cache_path}")
            pair = (context, mask)
            # Ordinary LIBERO repeatedly reuses a handful of instructions;
            # paper15 LIBERO-Plus mostly uses each prompt once.  A small FIFO
            # bound captures the former without retaining gigabytes in the
            # latter.
            if len(cache) >= 32:
                cache.pop(next(iter(cache)))
            cache[prompt] = pair

        context, mask = pair
        last_valid = int(mask.nonzero(as_tuple=False)[-1].item()) + 1
        return (
            context[:last_valid]
            .unsqueeze(0)
            .to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            mask[:last_valid]
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    @torch.no_grad()
    def _encode_ovis_u1_text(self, sample, condition_image: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                text_attention_mask = torch.ones(
                    cached_text_hidden_states.shape[:2],
                    device=cached_text_hidden_states.device,
                    dtype=torch.bool,
                )
            else:
                text_attention_mask = text_attention_mask.to(dtype=torch.bool)
            if (
                str(
                    getattr(
                        self,
                        "external_ssl_text_condition_source",
                        "ovis_multimodal",
                    )
                )
                == "qwen3_4b_flux2_cached"
            ):
                expected_hidden = int(self.external_ssl_qwen_hidden_dim)
                if int(cached_text_hidden_states.shape[-1]) != expected_hidden:
                    raise ValueError(
                        "DINO-native Qwen3 cached hidden width mismatch: "
                        f"{cached_text_hidden_states.shape[-1]} vs {expected_hidden}."
                    )
                valid_columns = text_attention_mask.any(dim=0)
                if not bool(valid_columns.any()):
                    raise ValueError("DINO-native Qwen3 batch has no valid text token.")
                trim_len = int(
                    valid_columns.nonzero(as_tuple=False)[-1].item()
                ) + 1
                cached_text_hidden_states = cached_text_hidden_states[:, :trim_len]
                text_attention_mask = text_attention_mask[:, :trim_len]
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = text_attention_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            return text_hidden_states, text_attention_mask
        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("prompt")
        if instruction is None:
            instruction = sample.get("task")
        if instruction is not None:
            if condition_image is None:
                raise ValueError("Ovis-U1 online condition encoding requires the condition/input image.")
            # IMAGEWAM_COND_CACHE_EMULATE=1 (measurement only): reuse the first batch's prompt
            # context instead of running the frozen visual tokenizer and language model again.
            if os.environ.get("JAM_COND_CACHE_EMULATE") == "1":
                cached = getattr(self, "_cond_emulate_context", None)
                if cached is not None and int(cached[0].shape[0]) == int(condition_image.shape[0]):
                    return cached
                out = self._encode_ovis_u1_condition_online(instruction, condition_image)
                self._cond_emulate_context = tuple(t.detach() for t in out)
                return out
            return self._encode_ovis_u1_condition_online(instruction, condition_image)
        raise ValueError(
            "Ovis-U1 training currently requires precomputed `text_hidden_states` "
            "from OvisU1.generate_condition(), or `instruction`/`prompt` with `load_condition_encoder=true`. "
            f"Available sample keys: {sorted(sample.keys())}"
        )

    @torch.no_grad()
    def _prepare_ovis_u1_infer_text(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            if (
                str(
                    getattr(
                        self,
                        "external_ssl_text_condition_source",
                        "ovis_multimodal",
                    )
                )
                == "qwen3_4b_flux2_cached"
            ):
                return self._read_external_ssl_qwen_text_cache(str(prompt))
            return self._encode_ovis_u1_condition_online(prompt, input_image)
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def _build_inputs_ovis_external_ssl(self, sample):
        """Build action-policy inputs from a trainable dense SSL student.

        The student sees only the current endpoint.  The raw future endpoint is
        retained for the unregistered frozen SSL teacher and is never passed to
        the student/Yak/action path.
        """

        from .ovis_u1_video_expert import OvisU1VideoExpert

        video = sample.get("video")
        current_frame = sample.get("current_frame", sample.get("input_image"))
        future_frame = sample.get("next_frame", sample.get("target_image"))
        ssl_fusion_type = str(
            getattr(
                self,
                "external_ssl_action_fusion_type",
                "layerwise_mot",
            )
        )
        native_videomae_action = (
            ssl_fusion_type == "native_videomae12_mot"
        )
        videomae_clip_teacher = (
            native_videomae_action
            and str(
                getattr(
                    self,
                    "external_ssl_teacher_input",
                    "real_future_clip_t1_to_t16_last_tubelet",
                )
            )
            == "real_future_clip_t1_to_t16_last_tubelet"
        )
        if video is not None:
            if video.ndim != 5:
                raise ValueError(
                    f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}"
                )
            if current_frame is None:
                current_frame = video[:, :, 0]
            if videomae_clip_teacher:
                expected_frames = 17
                if int(video.shape[2]) != expected_frames:
                    raise ValueError(
                        "Native VideoMAE policy requires current t plus the "
                        "real future clip t+1...t+16: expected T=17, got "
                        f"{tuple(video.shape)}."
                    )
                future_frame = video[:, :, 1:]
            elif future_frame is None:
                future_frame = video[:, :, -1]
        if current_frame is None or future_frame is None:
            raise ValueError(
                "External SSL Ovis training requires current and future endpoint images."
            )

        ssl_student = getattr(self.mot, "ssl_student", None)
        if ssl_student is None:
            raise RuntimeError("External SSL mode is missing the registered SSL student.")
        student_layer_features = None
        native_dino_action = ssl_fusion_type in {
            "native_dino12_mot",
            "native_dino24_mot",
            "native_dinov2_large24_mot",
            "native_videomae12_mot",
        }
        if native_dino_action:
            if current_frame.ndim not in (3, 4):
                raise ValueError(
                    "Native DINO action input must be [B,3,H,W] or [3,H,W], "
                    f"got {tuple(current_frame.shape)}."
                )
            batch_size = 1 if current_frame.ndim == 3 else int(current_frame.shape[0])
            # Native DINO performs its single visual pass together with prefix
            # K/V construction inside the policy forward.  A zero-length
            # placeholder preserves the common batch/device contract here and
            # prevents the old builder from running DINO a second time.
            student_features = torch.empty(
                batch_size,
                0,
                int(self.external_ssl_hidden_size),
                device=self.device,
                dtype=self.torch_dtype,
            )
        elif ssl_fusion_type == "direct_layerwise_kv":
            student_features, student_layer_features = ssl_student(
                current_frame,
                output_hidden_states=True,
            )
        else:
            student_features = ssl_student(current_frame)
        if student_features.ndim != 3:
            raise RuntimeError(
                f"SSL student must return [B,N,D], got {tuple(student_features.shape)}."
            )
        batch_size = int(student_features.shape[0])
        grid_h = int(self.external_ssl_grid_size)
        grid_w = grid_h * int(self.external_ssl_num_views)
        expected_tokens = grid_h * grid_w
        if not native_dino_action and int(student_features.shape[1]) != expected_tokens:
            raise RuntimeError(
                "SSL dense-grid mismatch: "
                f"expected {expected_tokens} tokens ({grid_h}x{grid_w}), "
                f"got {student_features.shape[1]}."
            )
        if student_layer_features is not None:
            expected_layers = int(self.external_ssl_num_layers)
            if len(student_layer_features) != expected_layers:
                raise RuntimeError(
                    "SSL hidden-state depth mismatch: "
                    f"expected {expected_layers}, got {len(student_layer_features)}."
                )
            invalid_shapes = [
                (layer_idx, tuple(features.shape))
                for layer_idx, features in enumerate(student_layer_features)
                if tuple(features.shape) != tuple(student_features.shape)
            ]
            if invalid_shapes:
                raise RuntimeError(
                    "SSL hidden-state dense-token mismatch: "
                    f"final={tuple(student_features.shape)} invalid={invalid_shapes}."
                )
        if native_dino_action:
            ref_img_ids = torch.empty(
                batch_size,
                0,
                3,
                device=student_features.device,
                dtype=student_features.dtype,
            )
        else:
            ref_img_ids = OvisU1VideoExpert.build_img_ids(
                batch_size=batch_size,
                token_height=grid_h,
                token_width=grid_w,
                time_value=1.0,
                device=student_features.device,
                dtype=student_features.dtype,
            )
        empty_target = student_features[:, :0]
        empty_target_ids = ref_img_ids[:, :0]

        text_hidden_states, text_attention_mask = self._encode_ovis_u1_text(
            sample,
            condition_image=current_frame,
        )
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="Ovis external-SSL training sample",
            )

        action = sample["action"].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is None:
            future_valid = torch.ones(
                batch_size,
                device=self.device,
                dtype=torch.bool,
            )
        else:
            image_is_pad = image_is_pad.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            if image_is_pad.ndim != 2 or int(image_is_pad.shape[0]) != batch_size:
                raise ValueError(
                    f"`image_is_pad` must be [B,T], got {tuple(image_is_pad.shape)}."
                )
            if videomae_clip_teacher:
                if int(image_is_pad.shape[1]) != 17:
                    raise ValueError(
                        "Native VideoMAE image padding mask must cover "
                        f"t...t+16, got {tuple(image_is_pad.shape)}."
                    )
                future_valid = ~image_is_pad[:, 1:].any(dim=1)
            else:
                future_valid = ~image_is_pad[:, -1]

        return {
            "target_latent": empty_target,
            "target_img_ids": empty_target_ids,
            "ref_image_latents": student_features,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "ssl_student_features": student_features,
            "ssl_student_layer_features": student_layer_features,
            "ssl_current_image": current_frame,
            "ssl_future_image": future_frame,
            "ssl_future_valid": future_valid,
        }

    def _adapt_external_ssl_layer_features(
        self,
        layer_features: tuple[torch.Tensor, ...],
        *,
        final_adapted_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Map native DINO hidden states into ActionDiTYak's 1536-D space."""

        adapter = getattr(self.video_expert, "external_ssl_img_in", None)
        if adapter is None:
            raise RuntimeError(
                "Layerwise DINO action fusion is missing its shared visual "
                "adapter."
            )
        expected_layers = int(self.external_ssl_num_layers)
        if len(layer_features) != expected_layers:
            raise RuntimeError(
                "Layerwise DINO feature depth mismatch: "
                f"expected {expected_layers}, got {len(layer_features)}."
            )
        adapted = [
            adapter(features)
            for features in layer_features[:-1]
        ]
        if final_adapted_features is None:
            adapted.append(adapter(layer_features[-1]))
        else:
            if tuple(final_adapted_features.shape[:2]) != tuple(
                layer_features[-1].shape[:2]
            ):
                raise RuntimeError(
                    "Final adapted DINO token shape mismatch: "
                    f"adapted={tuple(final_adapted_features.shape)} "
                    f"native={tuple(layer_features[-1].shape)}."
                )
            adapted.append(final_adapted_features)
        return tuple(adapted)

    def build_inputs_ovis_u1(self, sample, tiled: bool = False):
        if bool(getattr(self, "external_ssl_enabled", False)):
            return self._build_inputs_ovis_external_ssl(sample)

        video = sample.get("video")
        condition_image = None
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `next_frame`, `target_image`, or target token fields.")
            # IMAGEWAM_FORESIGHT_CACHE_EMULATE=1 (throughput measurement only): reuse the first
            # batch's future-frame tokens instead of re-encoding, as a target cache would supply them.
            cache_emulate = os.environ.get("JAM_FORESIGHT_CACHE_EMULATE") == "1"
            cached = getattr(self, "_cache_emulate_target_tokens", None)
            if cache_emulate and cached is not None and int(cached[0].shape[0]) == int(next_frame.shape[0]):
                target_tokens, target_img_ids = cached
            else:
                target_tokens, target_img_ids = self._encode_ovis_u1_image_tokens(next_frame, time_value=0.0)
                if cache_emulate:
                    self._cache_emulate_target_tokens = (target_tokens.detach(), target_img_ids.detach())

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("Ovis-U1 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            condition_image = current_frame
            ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(current_frame, time_value=1.0)

        if condition_image is None:
            condition_image = sample.get("current_frame", sample.get("input_image"))
            if condition_image is None and video is not None:
                condition_image = video[:, :, 0]
        text_hidden_states, text_attention_mask = self._encode_ovis_u1_text(sample, condition_image=condition_image)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="Ovis-U1 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        inputs = {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }
        if (
            bool(getattr(self, "foresight_enabled", False))
            and str(getattr(self, "foresight_teacher_type", "yak")) == "videomae_clip"
        ):
            if video is None or video.ndim != 5:
                raise ValueError(
                    "VideoMAE foresight teacher requires the full RGB window "
                    "`sample['video']` with shape [B,3,T,H,W]."
                )
            clip_frames = int(self.foresight_teacher_clip_frames)
            if int(video.shape[2]) != clip_frames + 1:
                raise ValueError(
                    "VideoMAE foresight expects current t plus the real future clip: "
                    f"required T={clip_frames + 1}, got {tuple(video.shape)}."
                )
            inputs["foresight_teacher_video"] = video[:, :, 1:]
            image_is_pad = sample.get("image_is_pad")
            if image_is_pad is None:
                clip_valid = torch.ones(
                    int(video.shape[0]),
                    device=self.device,
                    dtype=torch.bool,
                )
            else:
                if image_is_pad.ndim != 2 or tuple(image_is_pad.shape) != (
                    int(video.shape[0]),
                    clip_frames + 1,
                ):
                    raise ValueError(
                        "VideoMAE foresight `image_is_pad` must cover current+future "
                        f"[B,{clip_frames + 1}], got {tuple(image_is_pad.shape)}."
                    )
                clip_valid = ~image_is_pad[:, 1:].to(
                    device=self.device,
                    dtype=torch.bool,
                    non_blocking=True,
                ).any(dim=1)
            inputs["foresight_teacher_valid"] = clip_valid
        return inputs

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self,
        image: torch.Tensor,
        *,
        time_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}")
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _decode_flux2_image_tokens(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        from .flux2_video_expert import Flux2VideoExpert

        latent_h = int(height) // 16
        latent_w = int(width) // 16
        latents = Flux2VideoExpert.unpack_latents(tokens, latent_h, latent_w)
        image = self.vae.decode(latents.to(device=self.device, dtype=self.torch_dtype))
        return image.detach().float().clamp(-1, 1)

    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                raise ValueError("FLUX.2 cached `text_hidden_states` must be paired with `text_attention_mask`.")
            if cached_text_hidden_states.ndim != 3 or text_attention_mask.ndim != 2:
                raise ValueError(
                    "FLUX.2 cached context/mask must be [B,L,D]/[B,L], got "
                    f"{tuple(cached_text_hidden_states.shape)}/{tuple(text_attention_mask.shape)}."
                )
            if tuple(cached_text_hidden_states.shape[:2]) != tuple(text_attention_mask.shape):
                raise ValueError(
                    "FLUX.2 cached context/mask shape mismatch: "
                    f"{tuple(cached_text_hidden_states.shape)}/{tuple(text_attention_mask.shape)}."
                )
            text_attention_mask = text_attention_mask.to(dtype=torch.bool)
            if self.trim_cached_text_padding:
                # The official Qwen3 cache uses right padding.  Validate that
                # invariant rather than silently changing left- or hole-padded
                # inputs, then retain exactly the longest valid prefix in this
                # batch.  At least one token per sample is required by FLUX.
                lengths = text_attention_mask.sum(dim=1, dtype=torch.int64)
                if bool((lengths <= 0).any().item()):
                    raise ValueError("FLUX.2 cached text mask contains an empty sample.")
                positions = torch.arange(
                    text_attention_mask.shape[1],
                    device=text_attention_mask.device,
                ).unsqueeze(0)
                expected_mask = positions < lengths.unsqueeze(1)
                if not bool(torch.equal(text_attention_mask, expected_mask)):
                    raise ValueError(
                        "FLUX.2 padding trim requires contiguous right-padded cached masks."
                    )
                trim_len = int(lengths.max().item())
                cached_text_hidden_states = cached_text_hidden_states[:, :trim_len]
                text_attention_mask = text_attention_mask[:, :trim_len]
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            text_attention_mask = text_attention_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            return text_hidden_states, text_attention_mask

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "FLUX.2 stack requires precomputed `text_hidden_states` or an `instruction`/`prompt`/`task`."
            )
        if self.text_encoder is None:
            raise ValueError("FLUX.2 online text encoding requires `load_text_encoder=true`.")
        video = sample.get("video")
        batch_size = int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        return self._encode_flux2_prompts(prompts)

    @torch.no_grad()
    def _encode_flux2_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            self._load_flux2_text_encoder_for_inference()
        if not hasattr(self.text_encoder, "tokenizer") or not hasattr(self.text_encoder, "model"):
            text_hidden = self.text_encoder(list(prompts)).to(device=self.device, dtype=self.torch_dtype)
            text_mask = torch.ones(text_hidden.shape[:2], device=self.device, dtype=torch.bool)
            return text_hidden, text_mask

        ensure_flux2_importable_path = getattr(self, "model_paths", {}).get("flux2_src")
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(ensure_flux2_importable_path)
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        tokenizer = self.text_encoder.tokenizer
        model = self.text_encoder.model
        max_length = int(getattr(self.text_encoder, "max_length", 512))
        all_input_ids = []
        all_attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)")
        return (
            hidden.to(device=self.device, dtype=self.torch_dtype),
            attention_mask.to(device=self.device, dtype=torch.bool),
        )

    def _load_flux2_text_encoder_for_inference(self) -> None:
        from types import SimpleNamespace
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_spec = getattr(self, "flux2_qwen3_model_spec", None)
        if model_spec is None:
            model_spec = getattr(self, "model_paths", {}).get("qwen3")
        if model_spec is None:
            if int(self.text_dim) == 7680:
                model_spec = "Qwen/Qwen3-4B"
            elif int(self.text_dim) == 12288:
                model_spec = "Qwen/Qwen3-8B"
            else:
                raise ValueError(f"Cannot infer FLUX.2 Qwen3 model spec from text_dim={self.text_dim}")
        logger.info("Lazy-loading FLUX.2 Qwen3 text encoder for prompt inference: %s", model_spec)
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=self.torch_dtype,
        ).to(self.device).eval()
        self.text_encoder = SimpleNamespace(
            model=qwen3_model,
            tokenizer=AutoTokenizer.from_pretrained(model_spec),
            max_length=int(getattr(self, "qwen_context_len", 512)),
        )

    @torch.no_grad()
    def _prepare_flux2_infer_text(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self._encode_flux2_prompts([prompt])
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    def build_inputs_flux2(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if "target_latent" in sample and "target_img_ids" in sample:
            target_tokens = sample["target_latent"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            target_img_ids = sample["target_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            next_frame = sample.get("next_frame", sample.get("target_image"))
            if next_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                next_frame = video[:, :, -1]
            if next_frame is None:
                raise ValueError("FLUX.2 stack sample requires `next_frame`, `target_image`, or target token fields.")
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(next_frame, time_value=0.0)

        if "ref_image_latents" in sample and "ref_img_ids" in sample:
            ref_tokens = sample["ref_image_latents"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ref_img_ids = sample["ref_img_ids"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        else:
            current_frame = sample.get("current_frame", sample.get("input_image"))
            if current_frame is None and video is not None:
                if video.ndim != 5:
                    raise ValueError(f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}")
                current_frame = video[:, :, 0]
            if current_frame is None:
                raise ValueError("FLUX.2 stack sample requires `current_frame`, `input_image`, or ref token fields.")
            ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(current_frame, time_value=10.0)

        text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="FLUX.2 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        result = {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }
        return result

    def build_inputs_dim(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if video is None or video.ndim != 5:
            raise ValueError(f"DIM stack sample requires `video` [B,3,T,H,W], got {None if video is None else tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"DIM stack expects RGB video, got {tuple(video.shape)}")
        source_image = sample.get("current_frame", sample.get("input_image"))
        target_image = sample.get("next_frame", sample.get("target_image"))
        if source_image is None:
            source_image = video[:, :, 0]
        if target_image is None:
            target_image = video[:, :, -1]
        if source_image.shape[-2] % int(self.video_expert.vae_downsample_rate) != 0 or source_image.shape[-1] % int(self.video_expert.vae_downsample_rate) != 0:
            raise ValueError(
                "DIM image spatial dims must be multiples of "
                f"{self.video_expert.vae_downsample_rate}, got {tuple(source_image.shape[-2:])}."
            )
        source_image = source_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        target_image = target_image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        source_latent = self._encode_dim_image_latents(source_image)
        target_latent = self._encode_dim_image_latents(target_image)

        cached_dim_condition = sample.get("dim_condition_hidden_states")
        cached_dim_mask = sample.get("dim_condition_attention_mask")
        cached_text = sample.get("text_hidden_states")
        cached_text_mask = sample.get("text_attention_mask")
        if cached_dim_condition is not None and cached_dim_mask is not None:
            condition, condition_mask = (
                cached_dim_condition.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_dim_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        elif cached_text is not None and cached_text_mask is not None:
            condition, condition_mask = (
                cached_text.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                cached_text_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        else:
            instruction = sample.get("instruction", sample.get("prompt"))
            if instruction is None:
                raise ValueError("DIM stack requires `instruction`/`prompt` for online Qwen image+text conditioning.")
            condition, condition_mask = self._encode_dim_condition_online(instruction, source_image)
        if self.proprio_encoder is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=sample.get("proprio"),
                source="DIM training sample",
            )

        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        return {
            "target_latent": target_latent,
            "source_latent": source_latent,
            "condition": condition,
            "condition_mask": condition_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_omnigen2(
        self,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
        max_video_seq_len: int,
        action_seq_len: int,
        device: torch.device,
        l_effective_ref_img_len: list[list[int]],
    ) -> torch.Tensor:
        batch_size = len(seq_lengths)
        total_seq_len = int(max_video_seq_len) + int(action_seq_len)
        action_start = int(max_video_seq_len)
        mask = torch.zeros(batch_size, total_seq_len, total_seq_len, dtype=torch.bool, device=device)
        for i, (cap_len, video_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            prefix_len = cap_len + sum(l_effective_ref_img_len[i])

            # video -> video: text/ref prefix cannot attend noisy target, matching
            # wan22 first_frame_causal where first-frame tokens cannot see future tokens.
            mask[i, :prefix_len, :prefix_len] = True
            mask[i, prefix_len:video_len, :video_len] = True
            # action -> action
            mask[i, action_start:, action_start:] = True
            # action -> text/ref prefix only. Text is included here because
            # OmniGen2-style ActionDiT has no separate text cross-attention.
            mask[i, action_start:, :prefix_len] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_mask_ovis_u1(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        x0 = txt_len
        c0 = txt_len + target_len
        a0 = txt_len + target_len + cond_len
        total = a0 + action_len
        joint = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Text/condition are stable prefix states and do not attend noisy target.
        joint[:, t0:x0, t0:x0] = True
        joint[:, t0:x0, c0:a0] = True
        joint[:, c0:a0, t0:x0] = True
        joint[:, c0:a0, c0:a0] = True
        # Target/noisy image can use text, condition, and target image tokens.
        joint[:, x0:c0, t0:a0] = True
        # Action uses text, condition, and action self context, but not target/noisy image.
        joint[:, a0:total, t0:x0] = True
        joint[:, a0:total, c0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            joint[:, :, t0:x0] &= text_valid[:, None, :]

        image_total = target_len + cond_len + action_len
        image_action_start = target_len + cond_len
        image = torch.zeros(batch_size, image_total, image_total, dtype=torch.bool, device=device)
        image[:, :target_len, : target_len + cond_len] = True
        image[:, target_len:image_action_start, target_len:image_action_start] = True
        image[:, image_action_start:, target_len:image_total] = True
        return {"joint_double": joint, "image_double": image, "single": joint.clone()}

    @staticmethod
    def _slice_ovis_action_attention_mask(
        attention_mask: dict[str, torch.Tensor],
        *,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
    ) -> dict[str, torch.Tensor]:
        """Select action-query rows for cached-prefix Yak action forwards."""

        prefix_len = int(txt_len) + int(target_len) + int(cond_len)
        action_start = prefix_len
        image_prefix_len = int(target_len) + int(cond_len)
        image_action_start = image_prefix_len
        return {
            "joint_double": attention_mask["joint_double"][
                :,
                action_start : action_start + int(action_len),
                : prefix_len + int(action_len),
            ],
            "image_double": attention_mask["image_double"][
                :,
                image_action_start : image_action_start + int(action_len),
                : image_prefix_len + int(action_len),
            ],
            "single": attention_mask["single"][
                :,
                action_start : action_start + int(action_len),
                : prefix_len + int(action_len),
            ],
        }

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        x0 = txt_len + cond_len
        a0 = txt_len + cond_len + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        # Stable text/ref prefix cannot attend to noisy target/action tokens.
        mask[:, t0:r0, t0:x0] = True
        mask[:, r0:x0, t0:x0] = True
        # Target/noisy image uses stable prefix and target self-context.
        mask[:, x0:a0, t0:a0] = True
        # Action uses stable prefix and action self-context, not target/noisy image.
        mask[:, a0:total, t0:x0] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError(
                    "`text_attention_mask` must be [B,txt_len], "
                    f"got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_is_pad is not None:
            dim_valid = (~action_dim_is_pad).to(device=action_loss_dim.device, dtype=action_loss_dim.dtype)
            dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0).unsqueeze(1)
            action_loss_token = (action_loss_dim * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum
        else:
            action_loss_token = action_loss_dim.mean(dim=2)

        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _training_loss_omnigen2(self, sample, tiled: bool = False):
        debug_every = int(os.environ.get("JAM_DEBUG_OMNIGEN2_FORWARD_EVERY", "0") or "0")
        debug_threshold = float(os.environ.get("JAM_DEBUG_OMNIGEN2_FORWARD_THRESHOLD", "0") or "0")
        debug_forward = debug_every > 0 or debug_threshold > 0
        debug_step = int(getattr(self, "_omnigen2_forward_debug_step", 0)) + 1
        self._omnigen2_forward_debug_step = debug_step
        debug_segments: list[tuple[str, float]] = []
        debug_meta: dict[str, Any] = {}

        def _sync_and_mark(name: str, start_time: float) -> float:
            if debug_forward and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            now = time.perf_counter()
            if debug_forward:
                debug_segments.append((name, now - start_time))
            return now

        if debug_forward and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        inputs = self.build_inputs_omnigen2(sample, tiled=tiled)
        t0 = _sync_and_mark("build_inputs", t0)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = target_latent.shape[0]

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        t0 = _sync_and_mark("noise_sched", t0)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
        )
        t0 = _sync_and_mark("video_pre", t0)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        t0 = _sync_and_mark("action_pre", t0)

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            l_effective_ref_img_len=video_pre.get("l_effective_ref_img_len"),
            device=video_tokens.device,
        )
        if debug_forward:
            debug_meta = {
                "B": int(batch_size),
                "video_seq": int(video_tokens.shape[1]),
                "action_seq": int(action_tokens.shape[1]),
                "cap_min": int(min(video_pre["encoder_seq_lengths"])),
                "cap_max": int(max(video_pre["encoder_seq_lengths"])),
                "seq_min": int(min(video_pre["seq_lengths"])),
                "seq_max": int(max(video_pre["seq_lengths"])),
                "target_shape": tuple(target_latent.shape),
                "action_shape": tuple(action.shape),
            }
        t0 = _sync_and_mark("mask", t0)
        tokens_out = self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={"video": None, "action": None},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        t0 = _sync_and_mark("mot", t0)

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 2, 3))
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
        )
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        t0 = _sync_and_mark("post_loss", t0)
        if debug_forward:
            total_debug = sum(value for _, value in debug_segments)
            should_log = (debug_every > 0 and debug_step % debug_every == 0) or (
                debug_threshold > 0 and total_debug >= debug_threshold
            )
            phase_names = ["build_inputs", "noise_sched", "video_pre", "action_pre", "mask", "mot", "post_loss"]
            segment_dict = dict(debug_segments)
            durations = [float(segment_dict.get(name, 0.0)) for name in phase_names]
            arrivals = []
            cumulative = 0.0
            for value in durations:
                cumulative += value
                arrivals.append(cumulative)
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / (1024**3)
                if self.device.type == "cuda"
                else 0.0
            )
            meta_values = [
                float(debug_step),
                1.0 if should_log else 0.0,
                float(debug_meta.get("B", 0)),
                float(debug_meta.get("video_seq", 0)),
                float(debug_meta.get("action_seq", 0)),
                float(debug_meta.get("cap_min", 0)),
                float(debug_meta.get("cap_max", 0)),
                float(debug_meta.get("seq_min", 0)),
                float(debug_meta.get("seq_max", 0)),
                float(peak_gib),
            ]
            profile_values = durations + arrivals + meta_values
            self._last_omnigen2_forward_profile = {
                "phase_names": phase_names,
                "num_phases": len(phase_names),
            }
            self._last_omnigen2_forward_profile_tensor = torch.tensor(
                profile_values,
                device=loss_total.device,
                dtype=torch.float32,
            )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _training_loss_ovis_native_dino(self, inputs: dict):
        """Action flow-matching with a native SSL/action prefix.

        The DINO or single-frame VideoMAE visual stream is executed exactly
        once on the current image. Its per-layer patch K/V feed the matching
        action layer, while its final dense tokens are reused by the future
        foresight objective. No Yak transformer block or VAE image token is
        executed.
        """

        ssl_student = getattr(self.mot, "ssl_student", None)
        if ssl_student is None:
            raise RuntimeError("Native DINO action training is missing `mot.ssl_student`.")
        current_image = inputs.get("ssl_current_image")
        if current_image is None:
            raise RuntimeError("Native DINO action training is missing the current image.")

        action = inputs["action"]
        batch_size = int(action.shape[0])
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action,
            noise_action,
            timestep_action,
        )
        target_action = self.train_action_scheduler.training_target(
            action,
            noise_action,
            timestep_action,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(
                timestep_action,
                self.train_action_scheduler,
            ),
        )
        prefix_cache = self.action_expert.prefill_dino_prefix(
            ssl_student=ssl_student,
            images=current_image,
            text_hidden_states=inputs["text_hidden_states"],
            text_attention_mask=inputs["text_attention_mask"],
        )
        student_features = prefix_cache.get("final_dense")
        if student_features is None:
            raise RuntimeError(
                "ActionDiTDINO prefix cache did not return final DINO dense tokens."
            )
        expected_tokens = (
            int(self.external_ssl_grid_size)
            * int(self.external_ssl_grid_size)
            * int(self.external_ssl_num_views)
        )
        if tuple(student_features.shape) != (
            batch_size,
            expected_tokens,
            int(self.external_ssl_hidden_size),
        ):
            raise RuntimeError(
                "Native DINO final dense-token shape mismatch: expected "
                f"{(batch_size, expected_tokens, int(self.external_ssl_hidden_size))}, "
                f"got {tuple(student_features.shape)}."
            )
        inputs["ssl_student_features"] = student_features
        action_tokens = self.action_expert.forward_with_prefix_cache(
            action_pre=action_pre,
            prefix_cache=prefix_cache,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": 0.0,
            "loss_action": self.loss_lambda_action
            * float(loss_action.detach().item()),
        }
        if bool(getattr(self, "foresight_enabled", False)):
            loss_foresight = self._compute_ovis_foresight_loss(
                inputs=inputs,
                video_pre={},
                captured=None,
                batch_size=batch_size,
            )
            loss_total = loss_total + self.foresight_lambda * loss_foresight
            loss_dict["loss_foresight"] = self.foresight_lambda * float(
                loss_foresight.detach().item()
            )
        return loss_total, loss_dict

    def _training_loss_ovis_u1(self, sample, tiled: bool = False):
        inputs = self.build_inputs_ovis_u1(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        foresight_on = bool(getattr(self, "foresight_enabled", False))
        external_ssl_on = bool(getattr(self, "external_ssl_enabled", False))
        external_ssl_fusion = str(
            getattr(
                self,
                "external_ssl_action_fusion_type",
                "layerwise_mot",
            )
        )
        if (
            external_ssl_on
            and external_ssl_fusion
            in {
                "native_dino12_mot",
                "native_dino24_mot",
                "native_dinov2_large24_mot",
                "native_videomae12_mot",
            }
        ):
            return self._training_loss_ovis_native_dino(inputs)
        direct_ssl_action = (
            external_ssl_on
            and external_ssl_fusion
            in {"direct_static_kv", "direct_layerwise_kv"}
        )
        layerwise_dino_action = (
            external_ssl_on
            and external_ssl_fusion == "direct_layerwise_kv"
        )
        compute_video_loss = float(self.loss_lambda_video) != 0.0
        if external_ssl_on and compute_video_loss:
            raise ValueError(
                "The external SSL visual student has no image-generation head; "
                "set model.loss.lambda_video=0."
            )
        if compute_video_loss:
            noise_video = torch.randn_like(target_latent)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=target_latent.dtype,
            )
            noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)
        else:
            # Foresight-only: the generation objective is off, so the video
            # stream needs no diffusion noising — run it on the CLEAN target at
            # t=0. This matches the frozen teacher's conditions exactly and
            # mirrors the Wan recipe, where the aligned (first-frame)
            # tokens run under t=0 modulation via separated timesteps. The
            # aligned cond positions never attend target tokens, so this also
            # leaves the alignment computation bit-identical.
            timestep_video = torch.zeros(batch_size, device=self.device, dtype=target_latent.dtype)
            noisy_latent = target_latent
            target_video = None

        # foresight.drop_target_tokens: drop the clean target tokens from
        # the STUDENT's video stream entirely. Mask-verified: cond/text/action
        # positions never attend target tokens, so with the generation loss off
        # this is bit-exact for every used computation and saves ~40% of the
        # image-token compute (the same equivalence as first_frame_only in the
        # Wan backend).
        # The TEACHER still consumes the clean target (its features are the
        # alignment target); only the student goes future-blind.
        student_x = noisy_latent
        student_target_ids = inputs["target_img_ids"]
        if foresight_on and not compute_video_loss and bool(getattr(self, "foresight_drop_target", False)):
            student_x = noisy_latent[:, :0]
            student_target_ids = student_target_ids[:, :0]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=student_x,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=student_target_ids,
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        capture_yak_foresight = foresight_on and not external_ssl_on
        if capture_yak_foresight:
            self.mot._foresight_capture_layers = {int(self.foresight_layer)}
        if direct_ssl_action:
            if int(video_pre["target_len"]) != 0:
                raise RuntimeError(
                    "DINO direct action fusion requires an empty "
                    "future target stream."
                )
            expected_cond_len = (
                int(self.external_ssl_grid_size)
                * int(self.external_ssl_grid_size)
                * int(self.external_ssl_num_views)
            )
            if int(video_pre["cond_len"]) != expected_cond_len:
                raise RuntimeError(
                    "DINO direct action fusion received an unexpected visual "
                    f"token count: expected {expected_cond_len}, "
                    f"got {video_pre['cond_len']}."
                )
            image_layer_tokens = None
            action_layer_map = None
            if layerwise_dino_action:
                raw_layer_features = inputs.get("ssl_student_layer_features")
                if raw_layer_features is None:
                    raise RuntimeError(
                        "Layerwise DINO action fusion is missing student "
                        "hidden states."
                    )
                image_layer_tokens = (
                    self._adapt_external_ssl_layer_features(
                        raw_layer_features,
                        final_adapted_features=video_pre["tokens"]["img"],
                    )
                )
                action_layer_map = list(
                    self.external_ssl_action_layer_map
                )
            prefix_cache = self.mot.prefill_yak_direct_prefix_cache(
                prefix_tokens=video_pre["tokens"],
                prefix_freqs=video_pre["freqs"],
                prefix_t_mod=video_pre["t_mod"],
                image_layer_tokens=image_layer_tokens,
                action_layer_map=action_layer_map,
            )
            action_attention_mask = self._slice_ovis_action_attention_mask(
                attention_mask,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=int(action_pre["tokens"].shape[1]),
            )
            action_out = self.mot.forward_yak_action_with_direct_prefix_cache(
                action_tokens=action_pre["tokens"],
                action_ids=action_pre["ids"],
                action_t_mod=action_pre["t_mod"],
                prefix_kv_cache=prefix_cache,
                attention_mask=action_attention_mask,
            )
            tokens_out = {
                "video": video_pre["tokens"],
                "action": action_out,
            }
        else:
            tokens_out = self.mot(
                embeds_all={
                    "video": video_pre["tokens"],
                    "action": action_pre["tokens"],
                },
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"]},
                context_all={
                    "video": None,
                    "action": {"ids": action_pre["ids"]},
                },
                t_mod_all={
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                },
            )
        foresight_captured = None
        if capture_yak_foresight:
            foresight_captured = getattr(self.mot, "_foresight_video_hiddens", None)
            self.mot._foresight_capture_layers = None
            self.mot._foresight_video_hiddens = None

        # Foresight-only mode (lambda_video=0) skips the generation head+loss
        # entirely. NOTE: no zero-weighted "probe" here — 0.0*loss still
        # backprops 0*jacobian, and a single bf16 inf in that jacobian injects
        # NaN into the SHARED parameter grads (observed: step-2 corruption).
        # The generation head simply receives no gradient in this mode.
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre) if compute_video_loss else None
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if os.environ.get("JAM_DEBUG_MAG") == "1":
            def _mx(t):
                if not torch.is_tensor(t):
                    return "NA"
                a = t.detach().float().abs()
                fin = a[torch.isfinite(a)]
                return f"{fin.amax().item():.3g}" if fin.numel() else "ALLNONFIN"
            logger.error(
                "[MAG] tv=%.0f ta=%.0f | tgtlat=%s noisy=%s txt=%s ref=%s | vtok=%s atok=%s | outv=%s outa=%s | predv=%s preda=%s tgtv=%s tgta=%s",
                float(timestep_video.float().mean()), float(timestep_action.float().mean()),
                _mx(target_latent), _mx(noisy_latent), _mx(inputs.get("text_hidden_states")), _mx(inputs.get("ref_image_latents")),
                _mx(video_pre["tokens"]), _mx(action_pre["tokens"]),
                _mx(tokens_out["video"]), _mx(tokens_out["action"]),
                _mx(pred_video), _mx(pred_action), _mx(target_video), _mx(target_action),
            )

        if compute_video_loss:
            video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype,
            )
            loss_video = (video_loss_per_sample * video_weight).mean()
        else:
            loss_video = torch.zeros((), device=target_latent.device, dtype=torch.float32)
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if foresight_on:
            loss_foresight = self._compute_ovis_foresight_loss(
                inputs=inputs,
                video_pre=video_pre,
                captured=foresight_captured,
                batch_size=batch_size,
            )
            loss_total = loss_total + self.foresight_lambda * loss_foresight
            loss_dict["loss_foresight"] = self.foresight_lambda * float(
                loss_foresight.detach().item()
            )
            if not compute_video_loss and not external_ssl_on:
                # PARAMETER-level zero probe over every trainable dit param.
                # d(p.sum())/dp == 1 exactly, so 0.0 * probe adds an exact-zero
                # gradient to every param — params with real grads are
                # unchanged, and otherwise grad-less params (the unused
                # generation head) get genuine zero grads. Without this,
                # DeepSpeed's flat grad partition keeps uninitialized memory
                # for grad-less params and the global grad-norm / clipping
                # step goes NaN, corrupting ALL weights after step 1.
                # (An activation-path probe is NOT safe: 0 * inf-jacobian = NaN.)
                probe = torch.stack(
                    [p.float().sum() for p in self.mot.parameters() if p.requires_grad]
                ).sum()
                loss_total = loss_total + 0.0 * probe
        return loss_total, loss_dict

    def _compute_ovis_foresight_loss(
        self,
        inputs: dict,
        video_pre: dict,
        captured: Optional[dict],
        batch_size: int,
    ) -> torch.Tensor:
        """Wan-recipe foresight loss on the Ovis (Yak) stack.

        Student: cond (current-frame) token hiddens captured after
        ``foresight_layer`` inside the MoT forward.  The frozen target is
        either Yak features on clean endpoints or the final spatial tubelet of
        a VideoMAE encoding of the real future clip.  In both cases the teacher
        is detached and the objective is token-wise cosine alignment.
        """
        if bool(getattr(self, "external_ssl_enabled", False)):
            from .external_ssl import get_external_ssl_teacher

            student_features = inputs.get("ssl_student_features")
            future_image = inputs.get("ssl_future_image")
            future_valid = inputs.get("ssl_future_valid")
            if student_features is None or future_image is None or future_valid is None:
                raise RuntimeError("External SSL foresight inputs are incomplete.")
            teacher = get_external_ssl_teacher(self)
            if teacher is None:
                raise RuntimeError(
                    "External SSL foresight training requires the frozen teacher; "
                    "JAM_FORESIGHT_NO_TEACHER is inference-only."
                )
            with torch.no_grad():
                target = teacher(future_image)
            if tuple(target.shape) != tuple(student_features.shape):
                raise RuntimeError(
                    "External SSL student/teacher dense-token mismatch: "
                    f"student={tuple(student_features.shape)} teacher={tuple(target.shape)}."
                )
            projector = self.mot.foresight_projector
            prediction = projector.head(projector.trunk(student_features), 0)
            per_sample = (
                1.0
                - F.cosine_similarity(
                    prediction.float(),
                    target.detach().float(),
                    dim=-1,
                )
            ).mean(dim=1)
            valid = future_valid.to(device=per_sample.device, dtype=per_sample.dtype)
            if bool(valid.any().item()):
                return (per_sample * valid).sum() / valid.sum()
            return prediction.float().sum() * 0.0

        layer = int(self.foresight_layer)
        if not captured or layer not in captured:
            raise RuntimeError(f"Foresight capture missing for layer {layer}.")
        student_target_len = int(video_pre["target_len"])  # 0 when drop_target_tokens
        student_img = captured[layer]  # [B, student_target_len + cond_len, C]
        student_cond = student_img[:, student_target_len:, :]

        if str(getattr(self, "foresight_teacher_type", "yak")) == "vae_direct":
            from .foresight_yak import lift_vae_direct_tokens

            # `target_latent` is already the frozen pretrained Ovis VAE's
            # shift/scaling-normalized, 2x2-packed representation of t+16.
            # Reuse it directly: no second VAE pass and no Yak teacher.
            target = lift_vae_direct_tokens(
                inputs["target_latent"].detach(),
                output_dim=int(self.foresight_vae_direct_output_dim),
            )
            projector = self.mot.foresight_projector
            prediction = projector.head(projector.trunk(student_cond), 0)
            if tuple(prediction.shape) != tuple(target.shape):
                raise RuntimeError(
                    "Ovis student/VAE-direct target mismatch: "
                    f"student_prediction={tuple(prediction.shape)} "
                    f"vae_target={tuple(target.shape)}."
                )
            return cosine_align_loss(prediction, target)

        if str(getattr(self, "foresight_teacher_type", "yak")) == "videomae_clip":
            teacher_video = inputs.get("foresight_teacher_video")
            teacher_valid = inputs.get("foresight_teacher_valid")
            if teacher_video is None or teacher_valid is None:
                raise RuntimeError("VideoMAE foresight teacher inputs are incomplete.")
            teacher = getattr(self, "foresight_teacher", None)
            if teacher is None:
                raise RuntimeError(
                    "VideoMAE foresight training requires the frozen teacher; "
                    "JAM_FORESIGHT_NO_TEACHER is inference-only."
                )
            with torch.no_grad():
                target = teacher(teacher_video)
            projector = self.mot.foresight_projector
            prediction = projector.head(projector.trunk(student_cond), 0)
            if tuple(prediction.shape) != tuple(target.shape):
                raise RuntimeError(
                    "Ovis student/VideoMAE teacher token mismatch: "
                    f"student_prediction={tuple(prediction.shape)} "
                    f"teacher={tuple(target.shape)}."
                )
            per_sample = (
                1.0
                - F.cosine_similarity(
                    prediction.float(),
                    target.detach().float(),
                    dim=-1,
                )
            ).mean(dim=1)
            valid = teacher_valid.to(
                device=per_sample.device,
                dtype=per_sample.dtype,
            )
            if bool(valid.any().item()):
                return (per_sample * valid).sum() / valid.sum()
            return prediction.float().sum() * 0.0

        teacher = self.foresight_teacher
        # IMAGEWAM_FORESIGHT_CACHE_EMULATE=1 (throughput measurement only): skip the frozen
        # teacher pass after the first batch and reuse its features, as a target cache would.
        cache_emulate = os.environ.get("JAM_FORESIGHT_CACHE_EMULATE") == "1"
        cached_feats = getattr(self, "_cache_emulate_teacher_feats", None)
        if cache_emulate and cached_feats is not None and int(cached_feats[1].shape[0]) == int(batch_size):
            teacher_feats = cached_feats
        else:
          with torch.no_grad():
            t_pre = teacher.expert.pre_dit(
                x=inputs["target_latent"],
                timestep=torch.zeros(batch_size, device=self.device, dtype=inputs["target_latent"].dtype),
                context=inputs["text_hidden_states"],
                context_mask=inputs["text_attention_mask"],
                ref_image_hidden_states=inputs["ref_image_latents"],
                target_img_ids=inputs["target_img_ids"],
                ref_img_ids=inputs["ref_img_ids"],
            )
            # Teacher masks are built FRESH with action_len=0 (video-only) from
            # the teacher's own token layout — freshly built masks are
            # contiguous (fused attention kernels hit illegal memory accesses
            # on non-contiguous mask views) and independent of the student's
            # layout (which has no target tokens under drop_target_tokens).
            t_target_len = int(t_pre["target_len"])
            t_cond_len = int(t_pre["cond_len"])
            teacher_mask = self._build_mot_attention_mask_ovis_u1(
                batch_size=batch_size,
                txt_len=int(t_pre["txt_len"]),
                target_len=t_target_len,
                cond_len=t_cond_len,
                action_len=0,
                device=student_img.device,
                text_attention_mask=t_pre["text_mask"],
            )
            t_out = self.mot.forward_yak_video_only(
                video_tokens=t_pre["tokens"],
                video_freqs=t_pre["freqs"],
                video_t_mod=t_pre["t_mod"],
                attention_mask=teacher_mask,
                capture_layers={int(self.foresight_encoder_layer)},
                video_expert=teacher.expert,
                early_exit_after_last_capture=bool(self.foresight_teacher_early_exit),
            )
            t_img = t_out["captured"][int(self.foresight_encoder_layer)]
            # offset 0 = current frame (cond positions), 1 = future (target positions)
            teacher_feats = {
                0: t_img[:, t_target_len:, :],
                1: t_img[:, :t_target_len, :],
            }
          if cache_emulate:
            self._cache_emulate_teacher_feats = {k: v.detach() for k, v in teacher_feats.items()}

        projector = self.mot.foresight_projector
        trunk_out = projector.trunk(student_cond)
        losses = []
        for idx, offset in enumerate(self.foresight_offsets):
            pred = projector.head(trunk_out, idx)
            losses.append(cosine_align_loss(pred, teacher_feats[int(offset)].detach()))
        return torch.stack(losses).mean()

    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        foresight_enabled = bool(getattr(self, "foresight_enabled", False))
        # A zero-lambda post-foresight calibration keeps the registered
        # projector/checkpoint topology and future-blind student semantics, but
        # must not execute or allocate the frozen teacher.  Keep target dropping
        # separate from whether the auxiliary loss itself is active.
        foresight_on = foresight_enabled and float(self.foresight_lambda) != 0.0
        compute_video_loss = float(self.loss_lambda_video) != 0.0
        if compute_video_loss:
            noise_video = torch.randn_like(target_latent)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=target_latent.dtype,
            )
            noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)
        else:
            # With generation disabled, match the clean frozen teacher at t=0.
            timestep_video = torch.zeros(batch_size, device=self.device, dtype=target_latent.dtype)
            noisy_latent = target_latent
            target_video = None

        student_x = noisy_latent
        student_target_ids = inputs["target_img_ids"]
        if foresight_enabled and not compute_video_loss and bool(getattr(self, "foresight_drop_target", False)):
            # FLUX.2 orders image tokens as [reference/current, target/future].
            # Stable-prefix and action rows cannot attend target tokens, so the
            # future tokens are dead student compute in foresight-only mode.
            student_x = noisy_latent[:, :0]
            student_target_ids = student_target_ids[:, :0]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=student_x,
            timestep=self._scheduler_timestep_to_unit(timestep_video, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=student_target_ids,
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        if foresight_on:
            self.mot._foresight_capture_layers = {int(self.foresight_layer)}
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        foresight_captured = None
        if foresight_on:
            foresight_captured = getattr(self.mot, "_foresight_video_hiddens", None)
            self.mot._foresight_capture_layers = None
            self.mot._foresight_video_hiddens = None

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre) if compute_video_loss else None
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        if compute_video_loss:
            video_loss_per_sample = F.mse_loss(
                pred_video.float(), target_video.float(), reduction="none"
            ).flatten(1).mean(dim=1)
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype,
            )
            loss_video = (video_loss_per_sample * video_weight).mean()
        else:
            loss_video = torch.zeros((), device=target_latent.device, dtype=torch.float32)
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        # Keep FLUX metrics as detached device tensors.  Trainer converts them
        # only on log/timer steps when gather_metrics_every_step=false, avoiding
        # metric-only CUDA synchronizations without changing optimizer math.
        loss_dict = {
            "loss_video": self.loss_lambda_video * loss_video.detach(),
            "loss_action": self.loss_lambda_action * loss_action.detach(),
        }
        if foresight_on:
            loss_foresight = self._compute_flux2_foresight_loss(
                inputs=inputs,
                video_pre=video_pre,
                captured=foresight_captured,
                batch_size=batch_size,
            )
            loss_total = loss_total + self.foresight_lambda * loss_foresight
            loss_dict["loss_foresight"] = self.foresight_lambda * loss_foresight.detach()
            if not compute_video_loss:
                # Two registered video-only outputs are bypassed when
                # lambda_video=0:
                #   1) the generation final layer; and
                #   2) the last single-stream block's output projection.
                # The latter still computes visual Q/K/V for the final action
                # block, but its post-attention video output has no downstream
                # consumer.  Give both paths explicit zero gradients for
                # DeepSpeed without scanning all ~4.6B MoT parameters.
                # Selecting one scalar still reaches the whole Parameter's
                # AccumulateGrad node and materializes an all-zero dense grad
                # after multiplication by zero; it avoids a forward reduction
                # over these otherwise-unused tensors.
                zero_modules = [
                    getattr(self.video_expert.transformer, "final_layer", None),
                ]
                single_blocks = getattr(self.video_expert.transformer, "single_blocks", None)
                if single_blocks is not None and len(single_blocks) > 0:
                    zero_modules.append(getattr(single_blocks[-1], "linear2", None))
                zero_anchor = None
                seen_parameters = set()
                for module in zero_modules:
                    if module is None:
                        continue
                    for parameter in module.parameters():
                        if not parameter.requires_grad or id(parameter) in seen_parameters:
                            continue
                        seen_parameters.add(id(parameter))
                        term = parameter.reshape(-1)[0].float()
                        zero_anchor = term if zero_anchor is None else zero_anchor + term
                if zero_anchor is not None:
                    loss_total = loss_total + 0.0 * zero_anchor
        return loss_total, loss_dict

    def _compute_flux2_foresight_loss(
        self,
        inputs: dict,
        video_pre: dict,
        captured: Optional[dict],
        batch_size: int,
    ) -> torch.Tensor:
        """Align current-frame student tokens to clean FLUX.2 teacher tokens."""
        layer = int(self.foresight_layer)
        if not captured or layer not in captured:
            raise RuntimeError(f"FLUX.2 foresight capture missing for layer {layer}.")

        student_img = captured[layer]
        student_cond_len = int(video_pre["cond_len"])
        student_cond = student_img[:, :student_cond_len]
        if student_cond.shape[1] == 0:
            raise RuntimeError("FLUX.2 foresight student has no current/reference image tokens.")

        teacher = getattr(self, "foresight_teacher", None)
        if teacher is None:
            raise RuntimeError(
                "FLUX.2 foresight training requires a teacher; "
                "JAM_FORESIGHT_NO_TEACHER is inference-only."
            )
        with torch.no_grad():
            teacher_pre = teacher.expert.pre_dit(
                x=inputs["target_latent"],
                timestep=torch.zeros(batch_size, device=self.device, dtype=inputs["target_latent"].dtype),
                context=inputs["text_hidden_states"],
                context_mask=inputs["text_attention_mask"],
                ref_image_hidden_states=inputs["ref_image_latents"],
                target_img_ids=inputs["target_img_ids"],
                ref_img_ids=inputs["ref_img_ids"],
            )
            teacher_cond_len = int(teacher_pre["cond_len"])
            teacher_target_len = int(teacher_pre["target_len"])
            teacher_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(teacher_pre["txt_len"]),
                target_len=teacher_target_len,
                cond_len=teacher_cond_len,
                action_len=0,
                device=student_img.device,
                text_attention_mask=teacher_pre["text_mask"],
            )
            # Be explicit for fused SDPA: sliced/non-contiguous masks caused
            # illegal accesses in the earlier Ovis port.
            teacher_mask = {name: mask.contiguous() for name, mask in teacher_mask.items()}
            teacher_out = self.mot.forward_flux2_video_only(
                video_tokens=teacher_pre["tokens"],
                video_freqs=teacher_pre["freqs"],
                video_t_mod=teacher_pre["t_mod"],
                attention_mask=teacher_mask,
                capture_layers={int(self.foresight_encoder_layer)},
                video_expert=teacher.expert,
                early_exit_after_last_capture=bool(self.foresight_teacher_early_exit),
            )
            teacher_img = teacher_out["captured"][int(self.foresight_encoder_layer)]
            teacher_features = {
                0: teacher_img[:, :teacher_cond_len],
                1: teacher_img[:, teacher_cond_len : teacher_cond_len + teacher_target_len],
            }

        projector = self.mot.foresight_projector
        trunk_out = projector.trunk(student_cond)
        losses = []
        for head_idx, offset in enumerate(self.foresight_offsets):
            target = teacher_features[int(offset)].detach()
            if target.shape[:2] != student_cond.shape[:2]:
                raise RuntimeError(
                    "FLUX.2 foresight token-grid mismatch: "
                    f"student={tuple(student_cond.shape)}, offset={offset} target={tuple(target.shape)}."
                )
            prediction = projector.head(trunk_out, head_idx)
            losses.append(cosine_align_loss(prediction, target))
        return torch.stack(losses).mean()

    def _training_loss_dim(self, sample, tiled: bool = False):
        inputs = self.build_inputs_dim(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        source_latent = inputs["source_latent"]
        condition = inputs["condition"]
        condition_mask = inputs["condition_mask"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])

        noise_video = torch.randn_like(target_latent)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        noisy_latent = self.train_video_scheduler.add_noise(target_latent, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(target_latent, noise_video, timestep_video)

        cond_latent = torch.randn_like(target_latent)
        timestep_cond = torch.full(
            (batch_size,),
            float(self.train_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=target_latent.dtype,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=noisy_latent,
            timestep=timestep_video,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "cond": cond_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=None,
            freqs_all={},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                    "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
                },
                "cond": {
                    "context": cond_pre["context"],
                    "mask": cond_pre["context_mask"],
                    "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
                },
                "action": {
                    "context_tokens": cond_pre["context_tokens"],
                    "context_mask_bool": cond_pre["context_mask_bool"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "cond": cond_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        video_loss_per_sample = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").flatten(1).mean(dim=1)
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            video_loss_per_sample.device,
            dtype=video_loss_per_sample.dtype,
        )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    def _debug_register_finite_hooks(self):
        # Env-gated (IMAGEWAM_DEBUG_FINITE=1): log the first LEAF module whose
        # output goes non-finite while its inputs were finite = the NaN origin.
        if getattr(self, "_finite_hooks_registered", False):
            return
        if os.environ.get("JAM_DEBUG_FINITE") != "1":
            return
        self._finite_hooks_registered = True

        def _mk(mod_name):
            def _hook(mod, inp, out):
                def _fin(x):
                    return (not torch.is_tensor(x)) or bool(torch.isfinite(x).all())
                ins_ok = all(_fin(x) for x in inp)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for o in outs:
                    if torch.is_tensor(o) and o.numel() and not bool(torch.isfinite(o).all()):
                        mx = o.detach().abs().amax().float().item()
                        logger.error(
                            "[FINITE] nonfinite OUT module=%s type=%s in_finite=%s out_absmax=%s shape=%s",
                            mod_name, type(mod).__name__, ins_ok, mx, tuple(o.shape),
                        )
                        return
            return _hook

        def _mkb(mod_name):
            def _bhook(mod, gin, gout):
                def _fin(x):
                    return (not torch.is_tensor(x)) or bool(torch.isfinite(x).all())
                gout_ok = all(_fin(g) for g in gout)
                for g in gin:
                    if torch.is_tensor(g) and g.numel() and not bool(torch.isfinite(g).all()):
                        finite = g.detach()[torch.isfinite(g.detach())]
                        mx = finite.abs().amax().float().item() if finite.numel() else float("inf")
                        logger.error(
                            "[GRAD] nonfinite grad_IN module=%s type=%s gout_finite=%s gin_finite_absmax=%s",
                            mod_name, type(mod).__name__, gout_ok, mx,
                        )
                        return
            return _bhook

        def _mkp(p_name):
            def _phook(grad):
                if grad is not None and torch.is_tensor(grad) and grad.numel() and not bool(torch.isfinite(grad).all()):
                    fin = grad.detach()[torch.isfinite(grad.detach())]
                    mx = fin.abs().amax().float().item() if fin.numel() else float("inf")
                    ninf = int((grad.detach() == float("inf")).sum() + (grad.detach() == float("-inf")).sum())
                    nnan = int(torch.isnan(grad.detach()).sum())
                    logger.error(
                        "[PGRAD] nonfinite PARAM grad param=%s shape=%s n_inf=%s n_nan=%s finite_absmax=%s",
                        p_name, tuple(grad.shape), ninf, nnan, mx,
                    )
                return grad
            return _phook

        n = 0
        for mod_name, mod in self.named_modules():
            if len(list(mod.children())) == 0:
                mod.register_forward_hook(_mk(mod_name))
                mod.register_full_backward_hook(_mkb(mod_name))
                n += 1
        p = 0
        for p_name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(_mkp(p_name))
                p += 1
        logger.error("[FINITE] registered hooks on %d leaf modules + %d trainable params", n, p)

    def _maybe_register_grad_sanitizer(self):
        # bf16 param-grads can overflow to +/-inf under large LLM-conditioning
        # activations (Qwen massive activations ~4e4) on some GPU software stacks.
        # Zero out only the non-finite grad ELEMENTS at the autograd level
        # (deepspeed-compatible; finite elements pass through unchanged), so a
        # rare overflow no longer corrupts weights. Gated by IMAGEWAM_SANITIZE_GRAD.
        if getattr(self, "_grad_sanitizer_registered", False):
            return
        if os.environ.get("JAM_SANITIZE_GRAD", "0") != "1":
            return
        self._grad_sanitizer_registered = True

        def _sanitize(grad):
            if grad is None or not torch.is_tensor(grad):
                return grad
            if bool(torch.isfinite(grad).all()):
                return grad
            return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        n = 0
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(_sanitize)
                n += 1
        logger.warning("[SANITIZE] registered non-finite grad zeroing on %d trainable params", n)

    def training_loss(self, sample, tiled: bool = False):
        self._maybe_register_grad_sanitizer()
        self._debug_register_finite_hooks()
        if self.stack == "dim":
            return self._training_loss_dim(sample, tiled=tiled)
        if self.stack == "flux2":
            return self._training_loss_flux2(sample, tiled=tiled)
        if self.stack == "ovis_u1":
            loss, loss_dict = self._training_loss_ovis_u1(sample, tiled=tiled)
            # bf16 backward can overflow under large LLM-conditioning activations
            # on some stacks. Scaling the loss scales the WHOLE backward graph so
            # no bf16 gradient overflows; Adam(W) is ~scale-invariant so the update
            # magnitude is unchanged. loss_dict keeps the true (unscaled) values.
            _ls = os.environ.get("JAM_LOSS_SCALE")
            if _ls:
                loss = loss * float(_ls)
            return loss, loss_dict
        if self.stack == "omnigen2":
            return self._training_loss_omnigen2(sample, tiled=tiled)

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        profile_infer_timing: bool = False,
    ) -> dict[str, Any]:
        if self.stack == "dim":
            return self.infer_action_dim(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            return self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "ovis_u1":
            return self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "omnigen2":
            return self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                profile_infer_timing=profile_infer_timing,
            )

        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        patch_size = tuple(getattr(self.video_expert, "patch_size", (1, 1, 1)))
        condition_grid = (
            int(first_frame_latents.shape[3]) // int(patch_size[1]),
            int(first_frame_latents.shape[4]) // int(patch_size[2]),
        )
        self._configure_action_attention_capture(
            condition_slice=(0, video_seq_len),
            condition_grid=condition_grid,
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "imagewam_infer_action",
                "condition": "first_frame_latents",
                "input_size": [int(height), int(width)],
            },
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer_action_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM action inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM action inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        cond_latent = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        timestep_cond = torch.full(
            (batch_size,),
            float(self.infer_video_scheduler.num_train_timesteps),
            device=self.device,
            dtype=source_latent.dtype,
        )
        cond_pre = self.video_expert.pre_dit(
            x=cond_latent,
            timestep=timestep_cond,
            context=condition,
            context_mask=condition_mask,
            latents_condition=source_latent if self.dim_with_latents_condition else None,
        )
        cond_cache = self.mot.prefill_sana_condition_cache(
            cond_tokens=cond_pre["tokens"],
            cond_t_mod=cond_pre["t_mod"],
            cond_context_payload={
                "context": cond_pre["context"],
                "mask": cond_pre["context_mask"],
                "hw": (cond_pre["meta"]["token_height"], cond_pre["meta"]["token_width"]),
            },
        )
        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_context_payload = {
            "context_tokens": cond_pre["context_tokens"],
            "context_mask_bool": cond_pre["context_mask_bool"],
        }
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
            )
            action_tokens = self.mot.forward_sana_action_with_condition_cache(
                action_tokens=action_pre["tokens"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload=action_context_payload,
                condition_cache=cond_cache,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta, latents_action)
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer_video_dim(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        downsample = int(self.video_expert.vae_downsample_rate)
        if input_image.shape[-2] % downsample != 0 or input_image.shape[-1] % downsample != 0:
            raise ValueError(
                f"DIM input spatial dims must be multiples of {downsample}, got {tuple(input_image.shape[-2:])}."
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            condition = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            condition_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        else:
            if prompt is None:
                raise ValueError("DIM video inference requires `prompt` unless precomputed context is provided.")
            condition, condition_mask = self._encode_dim_condition_online(prompt, input_image)
        if self.proprio_encoder is not None or proprio is not None:
            condition, condition_mask = self._append_proprio_to_context_if_enabled(
                context=condition,
                context_mask=condition_mask,
                proprio=proprio,
                source="DIM video inference",
            )

        source_latent = self._encode_dim_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            source_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=condition,
                context_mask=condition_mask,
                latents_condition=source_latent if self.dim_with_latents_condition else None,
            )
            x = video_pre["tokens"]
            context_payload = {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
                "hw": (video_pre["meta"]["token_height"], video_pre["meta"]["token_width"]),
            }
            hw = tuple(context_payload["hw"])
            for layer_idx, block in enumerate(self.video_expert.blocks):
                state = self.mot._sana_video_io(block, x, video_pre["t_mod"], hw)
                attn = self.mot._sana_linear_attention(block.attn, state["q"], state["k"], state["v"])
                x = self.mot._sana_video_post(block, attn, state, context_payload)
            pred_video = self.video_expert.post_dit(x, video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_dim_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_dim_separate(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        action_out = self.infer_action_dim(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        video_out = self.infer_video_dim(
            prompt=prompt,
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        image = video_out["image"].detach().float().clamp(-1, 1)
        image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return {"action": action_out["action"], "video": [Image.fromarray(image)]}

    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        empty_target = ref_tokens.new_zeros(batch_size, 0, ref_tokens.shape[-1])
        empty_target_ids = ref_img_ids.new_zeros(batch_size, 0, ref_img_ids.shape[-1])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        video_timestep = torch.zeros((batch_size,), dtype=ref_tokens.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=video_timestep,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        prefix_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = self.mot.prefill_flux2_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_attention_mask,
        )
        full_attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"])
        self._configure_action_attention_capture(
            condition_slice=(int(video_pre["txt_len"]), int(video_pre["txt_len"]) + int(video_pre["cond_len"])),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=prefix_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "flux2_infer_action",
                "condition": "ref_image_tokens",
                "txt_len": int(video_pre["txt_len"]),
                "cond_len": int(video_pre["cond_len"]),
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=None,
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"ids": action_pre["ids"]},
                video_kv_cache=video_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=prefix_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    def _forward_flux2_video_only(
        self,
        video_pre: dict[str, Any],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        video_state = video_pre["tokens"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video tokens must be a dict with `txt` and `img` tensors.")

        txt = video_state["txt"]
        img = video_state["img"]
        video_freqs = video_pre["freqs"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        video_t_mod = video_pre["t_mod"]
        video_expert = self.video_expert

        def _flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

        from flux2.model import apply_rope

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            q, k, v, mlp, gate = block._qkv(video_stream, video_t_mod["single"])
            q, k = apply_rope(q, k, stream_pe)
            mixed = self.mot._mixed_attention(
                _flatten_heads(q),
                _flatten_heads(k),
                _flatten_heads(v),
                attention_mask["single"],
            )
            video_stream = block._out(video_stream, mixed, mlp, gate)

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"txt": txt, "img": img}

    @torch.no_grad()
    def infer_video_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .flux2_video_expert import Flux2VideoExpert

        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"FLUX.2 image spatial dims must be multiples of 16, got HxW=({height},{width})")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="FLUX.2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        latent_h = int(height) // 16
        latent_w = int(width) // 16

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_tokens.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = Flux2VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h,
            token_width=latent_w,
            time_value=0.0,
            device=self.device,
            dtype=self.torch_dtype,
        )

        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            attention_mask = self._build_mot_attention_mask_flux2(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=latents_video.device,
                text_attention_mask=video_pre["text_mask"],
            )
            tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
            pred_video = self.video_expert.post_dit(tokens_out, video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_flux2_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_action_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        profile_infer_timing: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        profile_segments: list[tuple[str, float]] = []
        profile_enabled = bool(profile_infer_timing)
        profile_last = 0.0

        def _sync_profile_device() -> None:
            if not profile_enabled:
                return
            device = torch.device(self.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def _mark_profile(name: str) -> None:
            nonlocal profile_last
            if not profile_enabled:
                return
            _sync_profile_device()
            now = time.perf_counter()
            profile_segments.append((name, now - profile_last))
            profile_last = now

        if profile_enabled:
            _sync_profile_device()
            profile_last = time.perf_counter()

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        _mark_profile("prepare_text_s")
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 action inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        _mark_profile("encode_image_latents_s")
        batch_size = int(ref_latent.shape[0])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        _mark_profile("sample_latents_s")

        timestep_video = torch.zeros((batch_size,), dtype=ref_latent.dtype, device=self.device)
        _mark_profile("prepare_video_timestep_s")
        video_pre = self.video_expert.pre_dit(
            x=None,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_latent,
        )
        _mark_profile("video_pre_dit_s")
        video_seq_len = int(video_pre["tokens"].shape[1])
        text_len = int(video_pre["encoder_seq_lengths"][0])
        ref_len = int(sum(video_pre["l_effective_ref_img_len"][0]))
        self._configure_action_attention_capture(
            condition_slice=(text_len, text_len + ref_len),
            condition_grid=(int(input_image.shape[-2]) // 16, int(input_image.shape[-1]) // 16),
            prefix_len=video_seq_len,
            action_len=int(latents_action.shape[1]),
            metadata={
                "stack": str(self.stack),
                "source": "omnigen2_infer_action",
                "condition": "ref_image_latents",
                "text_len": text_len,
                "ref_len": ref_len,
                "input_size": [int(input_image.shape[-2]), int(input_image.shape[-1])],
            },
        )
        attention_mask = self._build_mot_attention_mask_omnigen2(
            encoder_seq_lengths=video_pre["encoder_seq_lengths"],
            seq_lengths=video_pre["seq_lengths"],
            max_video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
            device=video_pre["tokens"].device,
        )
        _mark_profile("build_attention_mask_s")
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload=None,
            video_attention_mask=attention_mask[:, :video_seq_len, :video_seq_len],
        )
        _mark_profile("prefill_video_cache_s")

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        _mark_profile("build_action_schedule_s")
        action_predict_times: list[float] = []
        action_loop_start = 0.0
        timing_payload = None
        if profile_enabled:
            _sync_profile_device()
            action_loop_start = time.perf_counter()
        for action_step_idx, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            self._start_action_attention_capture_step(action_step_idx)
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            if profile_enabled:
                _sync_profile_device()
                action_predict_start = time.perf_counter()
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=text_hidden,
                context_mask=text_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            if profile_enabled:
                _sync_profile_device()
                action_predict_times.append(time.perf_counter() - action_predict_start)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        if profile_enabled:
            _sync_profile_device()
            action_denoise_loop_s = time.perf_counter() - action_loop_start
            profile_segments.append(("action_denoise_loop_s", action_denoise_loop_s))
            action_predict_total_s = sum(action_predict_times)
            action_predict_count = len(action_predict_times)
            action_predict_avg_s = action_predict_total_s / action_predict_count if action_predict_count > 0 else 0.0
            action_predict_min_s = min(action_predict_times) if action_predict_times else 0.0
            action_predict_max_s = max(action_predict_times) if action_predict_times else 0.0
            total_profiled_s = sum(value for _, value in profile_segments)
            timing_payload = {
                "segments": {name: float(value) for name, value in profile_segments},
                "action_horizon": int(action_horizon),
                "num_inference_steps": int(action_predict_count),
                "action_predict_total_s": float(action_predict_total_s),
                "action_predict_avg_s": float(action_predict_avg_s),
                "action_predict_min_s": float(action_predict_min_s),
                "action_predict_max_s": float(action_predict_max_s),
                "total_profiled_s": float(total_profiled_s),
            }

        result = {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
        if timing_payload is not None:
            result["timing"] = timing_payload
        return result

    def _infer_action_ovis_native_dino(
        self,
        *,
        input_image: torch.Tensor,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
    ) -> dict[str, Any]:
        """Infer actions with one reusable DINO/text prefix cache."""

        ssl_student = getattr(self.mot, "ssl_student", None)
        if ssl_student is None:
            raise RuntimeError("Native DINO action inference is missing `mot.ssl_student`.")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        batch_size = int(input_image.shape[0])
        prefix_cache = self.action_expert.prefill_dino_prefix(
            ssl_student=ssl_student,
            images=input_image,
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
        )

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        noise_batch_size = 1 if generator is not None else batch_size
        latents_action = torch.randn(
            (noise_batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        )
        if noise_batch_size == 1 and batch_size > 1:
            latents_action = latents_action.expand(batch_size, -1, -1).clone()
        latents_action = latents_action.to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype,
                device=self.device,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(
                    timestep_action,
                    self.infer_action_scheduler,
                ),
            )
            action_tokens = self.action_expert.forward_with_prefix_cache(
                action_pre=action_pre,
                prefix_cache=prefix_cache,
            )
            pred_action = self.action_expert.post_dit(
                action_tokens,
                action_pre,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                step_delta_action,
                latents_action,
            )
        action = latents_action.detach().to(device="cpu", dtype=torch.float32)
        return {"action": action[0] if batch_size == 1 else action}

    @torch.no_grad()
    def infer_action_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] < 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [B,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")

        profile_requested = os.environ.get("JAM_PROFILE_OVIS_ACTION", "0") == "1"
        profile_limit = int(os.environ.get("JAM_PROFILE_OVIS_ACTION_CALLS", "2"))
        profile_call = int(getattr(self, "_ovis_action_profile_calls", 0))
        profile_enabled = profile_requested and profile_call < profile_limit
        if profile_requested:
            self._ovis_action_profile_calls = profile_call + 1
        profile_segments: list[tuple[str, float]] = []
        profile_last = 0.0

        def _sync_profile_device() -> None:
            if profile_enabled and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))

        def _mark_profile(name: str) -> None:
            nonlocal profile_last
            if not profile_enabled:
                return
            _sync_profile_device()
            now = time.perf_counter()
            profile_segments.append((name, now - profile_last))
            profile_last = now

        if profile_enabled:
            _sync_profile_device()
            profile_last = time.perf_counter()

        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 action inference",
            )
        if (
            bool(getattr(self, "external_ssl_enabled", False))
            and str(
                getattr(
                    self,
                    "external_ssl_action_fusion_type",
                    "layerwise_mot",
                )
            )
            in {
                "native_dino12_mot",
                "native_dino24_mot",
                "native_dinov2_large24_mot",
                "native_videomae12_mot",
            }
        ):
            return self._infer_action_ovis_native_dino(
                input_image=input_image,
                text_hidden=text_hidden,
                text_mask=text_mask,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        _mark_profile("prepare_condition")
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        external_ssl_fusion = str(
            getattr(
                self,
                "external_ssl_action_fusion_type",
                "layerwise_mot",
            )
        )
        layerwise_dino_action = (
            bool(getattr(self, "external_ssl_enabled", False))
            and external_ssl_fusion == "direct_layerwise_kv"
        )
        ref_layer_features = None
        if bool(getattr(self, "external_ssl_enabled", False)):
            from .ovis_u1_video_expert import OvisU1VideoExpert

            ssl_student = getattr(self.mot, "ssl_student", None)
            if ssl_student is None:
                raise RuntimeError("External SSL inference is missing the SSL student.")
            if layerwise_dino_action:
                ref_tokens, ref_layer_features = ssl_student(
                    input_image,
                    output_hidden_states=True,
                )
            else:
                ref_tokens = ssl_student(input_image)
            ref_img_ids = OvisU1VideoExpert.build_img_ids(
                batch_size=int(ref_tokens.shape[0]),
                token_height=int(self.external_ssl_grid_size),
                token_width=int(self.external_ssl_grid_size)
                * int(self.external_ssl_num_views),
                time_value=1.0,
                device=ref_tokens.device,
                dtype=ref_tokens.dtype,
            )
            empty_input_dim = int(self.external_ssl_hidden_size)
        else:
            ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(
                input_image,
                time_value=1.0,
            )
            empty_input_dim = int(self.video_expert.transformer.in_channels)
        _mark_profile("encode_reference")
        batch_size = int(ref_tokens.shape[0])
        empty_target = torch.empty(
            batch_size,
            0,
            empty_input_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        empty_target_ids = torch.empty(batch_size, 0, 3, device=self.device, dtype=self.torch_dtype)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        noise_batch_size = 1 if generator is not None else batch_size
        latents_action = torch.randn(
            (noise_batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        )
        if noise_batch_size == 1 and batch_size > 1:
            # Sequential LIBERO evaluation recreates the same seeded generator
            # at every replan. Repeat that exact noise for each vectorized env
            # instead of consuming different offsets from one generator.
            latents_action = latents_action.expand(batch_size, -1, -1).clone()
        latents_action = latents_action.to(device=self.device, dtype=self.torch_dtype)
        timestep_video = torch.zeros((batch_size,), dtype=self.torch_dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=timestep_video,
            context=text_hidden,
            context_mask=text_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        _mark_profile("video_pre_dit")
        direct_ssl_action = (
            bool(getattr(self, "external_ssl_enabled", False))
            and external_ssl_fusion
            in {"direct_static_kv", "direct_layerwise_kv"}
        )
        if direct_ssl_action:
            image_layer_tokens = None
            action_layer_map = None
            if layerwise_dino_action:
                if ref_layer_features is None:
                    raise RuntimeError(
                        "Layerwise DINO inference is missing hidden states."
                    )
                image_layer_tokens = (
                    self._adapt_external_ssl_layer_features(
                        ref_layer_features,
                        final_adapted_features=video_pre["tokens"]["img"],
                    )
                )
                action_layer_map = list(
                    self.external_ssl_action_layer_map
                )
            video_kv_cache = self.mot.prefill_yak_direct_prefix_cache(
                prefix_tokens=video_pre["tokens"],
                prefix_freqs=video_pre["freqs"],
                prefix_t_mod=video_pre["t_mod"],
                image_layer_tokens=image_layer_tokens,
                action_layer_map=action_layer_map,
            )
            _mark_profile("build_direct_prefix_cache")
        else:
            prefix_attention_mask = self._build_mot_attention_mask_ovis_u1(
                batch_size=batch_size,
                txt_len=int(video_pre["txt_len"]),
                target_len=int(video_pre["target_len"]),
                cond_len=int(video_pre["cond_len"]),
                action_len=0,
                device=self.device,
                text_attention_mask=video_pre["text_mask"],
            )
            _mark_profile("build_prefix_mask")
            video_kv_cache = self.mot.prefill_yak_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                attention_mask=prefix_attention_mask,
            )
        _mark_profile("prefill_video_cache")
        full_attention_mask = self._build_mot_attention_mask_ovis_u1(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=self.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = (
            int(video_pre["txt_len"])
            + int(video_pre["target_len"])
            + int(video_pre["cond_len"])
        )
        cond_len = int(video_pre["cond_len"])
        action_len = int(latents_action.shape[1])
        action_attention_mask = self._slice_ovis_action_attention_mask(
            full_attention_mask,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=cond_len,
            action_len=action_len,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        _mark_profile("prepare_action")
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.expand(batch_size).to(dtype=latents_action.dtype, device=self.device)
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(timestep_action, self.infer_action_scheduler),
            )
            if direct_ssl_action:
                action_tokens = (
                    self.mot.forward_yak_action_with_direct_prefix_cache(
                        action_tokens=action_pre["tokens"],
                        action_ids=action_pre["ids"],
                        action_t_mod=action_pre["t_mod"],
                        prefix_kv_cache=video_kv_cache,
                        attention_mask=action_attention_mask,
                    )
                )
            else:
                action_tokens = self.mot.forward_action_with_video_cache(
                    action_tokens=action_pre["tokens"],
                    action_freqs=None,
                    action_t_mod=action_pre["t_mod"],
                    action_context_payload={"ids": action_pre["ids"]},
                    video_kv_cache=video_kv_cache,
                    attention_mask=action_attention_mask,
                    video_seq_len=prefix_len,
                )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        _mark_profile("action_denoise")
        action = latents_action.detach().to(device="cpu", dtype=torch.float32)
        result = {"action": action[0] if batch_size == 1 else action}
        _mark_profile("copy_action_cpu")
        if profile_enabled:
            logger.info(
                "Ovis action profile call=%d external_ssl=%s text=%s ref=%s "
                "steps=%d student_device=%s segments=%s total=%.4fs",
                profile_call + 1,
                bool(getattr(self, "external_ssl_enabled", False)),
                tuple(text_hidden.shape),
                tuple(ref_tokens.shape),
                int(num_inference_steps),
                (
                    str(next(self.mot.ssl_student.parameters()).device)
                    if bool(getattr(self, "external_ssl_enabled", False))
                    else "n/a"
                ),
                {name: round(value, 4) for name, value in profile_segments},
                sum(value for _, value in profile_segments),
            )
        return result

    @torch.no_grad()
    def infer_video_ovis_u1(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        from .ovis_u1_video_expert import OvisU1VideoExpert

        self.eval()
        if bool(getattr(self, "external_ssl_enabled", False)):
            raise RuntimeError(
                "The external-SSL Ovis policy is action-only and has no image-generation path."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`input_image` spatial dims must be multiples of 16, got HxW=({height},{width})")
        text_hidden, text_mask = self._prepare_ovis_u1_infer_text(prompt, input_image, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="Ovis-U1 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_ovis_u1_image_tokens(input_image, time_value=1.0)
        batch_size = int(ref_tokens.shape[0])
        latent_h = 2 * ((int(height) + 15) // 16)
        latent_w = 2 * ((int(width) + 15) // 16)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, (latent_h // 2) * (latent_w // 2), int(self.video_expert.transformer.in_channels)),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        target_img_ids = OvisU1VideoExpert.build_img_ids(
            batch_size=batch_size,
            token_height=latent_h // 2,
            token_width=latent_w // 2,
            time_value=0.0,
            device=self.device,
            dtype=self.torch_dtype,
        )
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=self._scheduler_timestep_to_unit(timestep_video, self.infer_video_scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            pred_video = self.video_expert.native_forward_from_pre_state(video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_ovis_u1_image_tokens(latents_video, height=height, width=width)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer_video_omnigen2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_omnigen2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="OmniGen2 video inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_latent = self._encode_omnigen2_image_latents(input_image)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            ref_latent.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        video_only_mot = MoT({"video": self.video_expert}, mot_checkpoint_mixed_attn=False).to(self.device)
        infer_timesteps, infer_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(ref_latent.shape[0]).to(dtype=latents_video.dtype, device=self.device)
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_latent,
            )
            attention_mask = self._build_mot_attention_mask_omnigen2(
                encoder_seq_lengths=video_pre["encoder_seq_lengths"],
                seq_lengths=video_pre["seq_lengths"],
                max_video_seq_len=video_pre["tokens"].shape[1],
                action_seq_len=0,
                l_effective_ref_img_len=video_pre["l_effective_ref_img_len"],
                device=video_pre["tokens"].device,
            )
            tokens_out = video_only_mot(
                embeds_all={"video": video_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"]},
                context_all={"video": None},
                t_mod_all={"video": video_pre["t_mod"]},
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta, latents_video)

        image = self._decode_omnigen2_image_latents(latents_video)
        return {"image": image[0].detach().cpu()}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        if self.stack == "ovis_u1":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for Ovis-U1 action inference.")
            action_out = self.infer_action_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            video_out = self.infer_video_ovis_u1(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "omnigen2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for OmniGen2 action inference.")
            action_out = self.infer_action_omnigen2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            video_out = self.infer_video_omnigen2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {
                "action": action_out["action"],
                "video": [Image.fromarray(image)],
            }
        if self.stack == "dim":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for DIM action inference.")
            return self.infer_dim_separate(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if self.stack == "flux2":
            if action_horizon is None:
                raise ValueError("`action_horizon` is required for FLUX.2 action inference.")
            action_out = self.infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            video_out = self.infer_video_flux2(
                prompt=prompt,
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
            image = video_out["image"].detach().float().clamp(-1, 1)
            image = ((image + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            return {"action": action_out["action"], "video": [Image.fromarray(image)]}

        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        if bool(getattr(self, "save_lora_merged", False)):
            from .lora import lora_merged_state_dict

            mot_state = lora_merged_state_dict(self.mot)
            checkpoint_format = "lora_merged"
        elif bool(getattr(self, "save_trainable_only", False)):
            trainable_names = {name for name, param in self.mot.named_parameters() if param.requires_grad}
            mot_state = {
                key: value
                for key, value in self.mot.state_dict().items()
                if key in trainable_names
            }
            checkpoint_format = "trainable_only"
        else:
            mot_state = self.mot.state_dict()
            checkpoint_format = "full"
        payload = {
            "mot": mot_state,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if bool(getattr(self, "external_ssl_enabled", False)):
            payload["external_ssl"] = {
                "backend": str(getattr(self, "external_ssl_backend", "huggingface")),
                "model_id": str(self.external_ssl_model_id),
                "image_size": int(self.external_ssl_image_size),
                "num_views": int(self.external_ssl_num_views),
                "grid_size": int(self.external_ssl_grid_size),
                "hidden_size": int(self.external_ssl_hidden_size),
                "num_layers": int(self.external_ssl_num_layers),
                "action_fusion_type": str(
                    getattr(
                        self,
                        "external_ssl_action_fusion_type",
                        "layerwise_mot",
                    )
                ),
                "action_layer_map": [
                    int(index)
                    for index in getattr(
                        self,
                        "external_ssl_action_layer_map",
                        [],
                    )
                ],
                "action_num_layers": int(
                    getattr(
                        self,
                        "external_ssl_action_num_layers",
                        len(self.action_expert.blocks),
                    )
                ),
                "hidden_state_source": str(
                    getattr(
                        self,
                        "external_ssl_hidden_state_source",
                        "final_only",
                    )
                ),
                "hidden_state_norm": str(
                    getattr(
                        self,
                        "external_ssl_hidden_state_norm",
                        "not_applicable",
                    )
                ),
                "layer_map_algorithm": str(
                    getattr(
                        self,
                        "external_ssl_layer_map_algorithm",
                        "not_applicable",
                    )
                ),
                "dense_token_selection": str(
                    getattr(
                        self,
                        "external_ssl_dense_token_selection",
                        "trailing_patch_grid",
                    )
                ),
                "student_input": str(
                    getattr(
                        self,
                        "external_ssl_student_input",
                        "current_endpoint_t",
                    )
                ),
                "teacher_input": str(
                    getattr(
                        self,
                        "external_ssl_teacher_input",
                        "future_endpoint_t16",
                    )
                ),
                "future_only": True,
                "text_condition_source": str(
                    getattr(
                        self,
                        "external_ssl_text_condition_source",
                        "ovis_multimodal",
                    )
                ),
            }
            if (
                payload["external_ssl"]["text_condition_source"]
                == "qwen3_4b_flux2_cached"
            ):
                payload["external_ssl"].update(
                    {
                        "text_context_len": int(
                            self.external_ssl_qwen_context_len
                        ),
                        "text_hidden_dim": int(
                            self.external_ssl_qwen_hidden_dim
                        ),
                        "text_cache_format": "qwen3_flux2",
                    }
                )
            if (
                str(
                    getattr(
                        self,
                        "external_ssl_action_fusion_type",
                        "layerwise_mot",
                    )
                )
                in {
                    "native_dino12_mot",
                    "native_dino24_mot",
                    "native_dinov2_large24_mot",
                    "native_videomae12_mot",
                }
            ):
                payload["external_ssl"]["native_action"] = {
                    "class": str(self.external_ssl_native_action_class),
                    "num_layers": int(self.external_ssl_action_num_layers),
                    "hidden_size": int(
                        self.external_ssl_native_action_hidden_size
                    ),
                    "num_heads": int(
                        self.external_ssl_native_action_num_heads
                    ),
                    "head_dim": int(
                        self.external_ssl_native_action_head_dim
                    ),
                    "mlp_hidden_size": int(
                        self.external_ssl_native_action_mlp_hidden_size
                    ),
                    "text_dim": int(
                        self.external_ssl_native_action_text_dim
                    ),
                    "max_action_horizon": int(
                        self.external_ssl_native_action_max_horizon
                    ),
                    "visual_kv": str(
                        self.external_ssl_native_visual_kv_source
                    ),
                    "text_prefix": str(
                        self.external_ssl_native_text_prefix
                    ),
                    "initialization": str(
                        self.external_ssl_native_action_initialization
                    ),
                    "position": str(
                        self.external_ssl_native_action_position
                    ),
                    "time_factor": float(
                        self.external_ssl_native_action_time_factor
                    ),
                    "visual_hidden_size": int(
                        self.external_ssl_native_visual_hidden_size
                    ),
                    "visual_num_layers": int(
                        self.external_ssl_native_visual_num_layers
                    ),
                    "action_layer_map": [
                        int(index)
                        for index in self.external_ssl_native_action_layer_map
                    ],
                    "parameter_count": int(
                        self.external_ssl_native_action_parameter_count
                    ),
                    "camera_encoding": str(
                        self.external_ssl_native_camera_encoding
                    ),
                    "action_visual_tokens": str(
                        self.external_ssl_native_action_visual_tokens
                    ),
                }
        if (
            bool(getattr(self, "foresight_enabled", False))
            and str(getattr(self, "foresight_teacher_type", "yak"))
            == "videomae_clip"
        ):
            payload["foresight_teacher"] = {
                "type": "videomae_clip",
                "model_id": str(self.foresight_teacher_model_id),
                "mode": str(self.foresight_teacher_mode),
                "random_seed": int(self.foresight_teacher_random_seed),
                "clip_frames": int(self.foresight_teacher_clip_frames),
                "grid_size": int(self.foresight_teacher_grid_size),
                "hidden_size": int(self.foresight_teacher_hidden_dim),
                "student_layer": int(self.foresight_layer),
                "offsets": [int(offset) for offset in self.foresight_offsets],
                "temporal_target": str(self.foresight_teacher_temporal_target),
                "future_only": True,
            }
        if checkpoint_format != "full":
            payload["checkpoint_format"] = checkpoint_format
            payload["save_trainable_only"] = bool(getattr(self, "save_trainable_only", False))
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, mmap: bool = False):
        load_kwargs = {"map_location": "cpu"}
        if mmap:
            load_kwargs["mmap"] = True
        payload = torch.load(path, **load_kwargs)
        logger.info("Loading JAMPolicy checkpoint from %s with payload keys=%s step=%s", path, sorted(payload.keys()), payload.get("step"))
        saved_ssl = payload.get("external_ssl")
        current_ssl = bool(getattr(self, "external_ssl_enabled", False))
        if saved_ssl is not None and not current_ssl:
            raise ValueError(
                "Checkpoint contains an external SSL student but the current model does not."
            )
        if saved_ssl is None and current_ssl:
            raise ValueError(
                "Current model expects an external SSL student but the checkpoint has no SSL metadata."
            )
        if saved_ssl is not None:
            saved_backend = str(saved_ssl.get("backend", "huggingface"))
            current_backend = str(
                getattr(self, "external_ssl_backend", "huggingface")
            )
            if saved_backend != current_backend:
                raise ValueError(
                    "External SSL checkpoint/model backend mismatch: "
                    f"saved={saved_backend!r}, current={current_backend!r}."
                )
            expected_ssl = {
                "image_size": int(self.external_ssl_image_size),
                "num_views": int(self.external_ssl_num_views),
                "grid_size": int(self.external_ssl_grid_size),
                "hidden_size": int(self.external_ssl_hidden_size),
            }
            mismatches = {
                key: (saved_ssl.get(key), value)
                for key, value in expected_ssl.items()
                if int(saved_ssl.get(key, -1)) != value
            }
            if mismatches:
                raise ValueError(
                    f"External SSL checkpoint/model topology mismatch: {mismatches}."
                )
            saved_text_source = str(
                saved_ssl.get("text_condition_source", "ovis_multimodal")
            )
            current_text_source = str(
                getattr(
                    self,
                    "external_ssl_text_condition_source",
                    "ovis_multimodal",
                )
            )
            if saved_text_source != current_text_source:
                raise ValueError(
                    "External SSL checkpoint/model text condition mismatch: "
                    f"saved={saved_text_source!r}, current={current_text_source!r}."
                )
            if current_text_source == "qwen3_4b_flux2_cached":
                expected_text = {
                    "text_context_len": int(
                        self.external_ssl_qwen_context_len
                    ),
                    "text_hidden_dim": int(
                        self.external_ssl_qwen_hidden_dim
                    ),
                    "text_cache_format": "qwen3_flux2",
                }
                text_mismatches = {
                    key: (saved_ssl.get(key), value)
                    for key, value in expected_text.items()
                    if saved_ssl.get(key) != value
                }
                if text_mismatches:
                    raise ValueError(
                        "External SSL Qwen3 checkpoint/model mismatch: "
                        f"{text_mismatches}."
                    )
            saved_student_input = str(
                saved_ssl.get("student_input", "current_endpoint_t")
            )
            saved_teacher_input = str(
                saved_ssl.get("teacher_input", "future_endpoint_t16")
            )
            current_student_input = str(
                getattr(
                    self,
                    "external_ssl_student_input",
                    "current_endpoint_t",
                )
            )
            current_teacher_input = str(
                getattr(
                    self,
                    "external_ssl_teacher_input",
                    "future_endpoint_t16",
                )
            )
            if (
                saved_student_input != current_student_input
                or saved_teacher_input != current_teacher_input
            ):
                raise ValueError(
                    "External SSL checkpoint/model input semantics mismatch: "
                    f"student={saved_student_input!r}/{current_student_input!r}, "
                    f"teacher={saved_teacher_input!r}/{current_teacher_input!r}."
                )
            saved_fusion = str(
                saved_ssl.get("action_fusion_type", "layerwise_mot")
            )
            current_fusion = str(
                getattr(
                    self,
                    "external_ssl_action_fusion_type",
                    "layerwise_mot",
                )
            )
            if saved_fusion != current_fusion:
                raise ValueError(
                    "External SSL checkpoint/model action fusion mismatch: "
                    f"saved={saved_fusion!r}, current={current_fusion!r}."
                )
            if current_fusion == "direct_layerwise_kv":
                saved_num_layers = int(saved_ssl.get("num_layers", -1))
                current_num_layers = int(self.external_ssl_num_layers)
                saved_layer_map = [
                    int(index)
                    for index in saved_ssl.get("action_layer_map", [])
                ]
                current_layer_map = [
                    int(index)
                    for index in self.external_ssl_action_layer_map
                ]
                expected_layerwise = {
                    "action_num_layers": int(
                        self.external_ssl_action_num_layers
                    ),
                    "hidden_state_source": str(
                        self.external_ssl_hidden_state_source
                    ),
                    "hidden_state_norm": str(
                        self.external_ssl_hidden_state_norm
                    ),
                    "layer_map_algorithm": str(
                        self.external_ssl_layer_map_algorithm
                    ),
                    "dense_token_selection": str(
                        self.external_ssl_dense_token_selection
                    ),
                }
                layerwise_mismatches = {
                    key: (saved_ssl.get(key), value)
                    for key, value in expected_layerwise.items()
                    if saved_ssl.get(key) != value
                }
                if (
                    saved_num_layers != current_num_layers
                    or saved_layer_map != current_layer_map
                    or layerwise_mismatches
                ):
                    raise ValueError(
                        "External SSL layerwise checkpoint/model topology "
                        "mismatch: "
                        f"layers={saved_num_layers}/{current_num_layers}, "
                        f"map={saved_layer_map}/{current_layer_map}, "
                        f"metadata={layerwise_mismatches}."
                    )
            elif current_fusion in {
                "native_dino12_mot",
                "native_dino24_mot",
                "native_dinov2_large24_mot",
                "native_videomae12_mot",
            }:
                saved_native = saved_ssl.get("native_action")
                if not isinstance(saved_native, dict):
                    raise ValueError(
                        "Native DINO action checkpoint is missing strict "
                        "`external_ssl.native_action` metadata."
                    )
                # The original native_dino12_mot checkpoints predate these
                # explicit capacity fields. Their values are fully implied by
                # the already-validated 12x768 topology, so fill them only for
                # backward-compatible loading. New 24x1536 checkpoints write
                # and validate all fields directly.
                saved_native = dict(saved_native)
                saved_native.setdefault(
                    "visual_hidden_size",
                    int(self.external_ssl_native_visual_hidden_size),
                )
                saved_native.setdefault(
                    "visual_num_layers",
                    int(self.external_ssl_native_visual_num_layers),
                )
                saved_native.setdefault(
                    "action_layer_map",
                    [
                        int(index)
                        for index in self.external_ssl_native_action_layer_map
                    ],
                )
                saved_native.setdefault(
                    "parameter_count",
                    int(self.external_ssl_native_action_parameter_count),
                )
                expected_native = {
                    "class": str(self.external_ssl_native_action_class),
                    "num_layers": int(self.external_ssl_action_num_layers),
                    "hidden_size": int(
                        self.external_ssl_native_action_hidden_size
                    ),
                    "num_heads": int(
                        self.external_ssl_native_action_num_heads
                    ),
                    "head_dim": int(
                        self.external_ssl_native_action_head_dim
                    ),
                    "mlp_hidden_size": int(
                        self.external_ssl_native_action_mlp_hidden_size
                    ),
                    "text_dim": int(
                        self.external_ssl_native_action_text_dim
                    ),
                    "max_action_horizon": int(
                        self.external_ssl_native_action_max_horizon
                    ),
                    "visual_kv": str(
                        self.external_ssl_native_visual_kv_source
                    ),
                    "text_prefix": str(
                        self.external_ssl_native_text_prefix
                    ),
                    "initialization": str(
                        self.external_ssl_native_action_initialization
                    ),
                    "position": str(
                        self.external_ssl_native_action_position
                    ),
                    "time_factor": float(
                        self.external_ssl_native_action_time_factor
                    ),
                    "visual_hidden_size": int(
                        self.external_ssl_native_visual_hidden_size
                    ),
                    "visual_num_layers": int(
                        self.external_ssl_native_visual_num_layers
                    ),
                    "action_layer_map": [
                        int(index)
                        for index in self.external_ssl_native_action_layer_map
                    ],
                    "parameter_count": int(
                        self.external_ssl_native_action_parameter_count
                    ),
                    "camera_encoding": str(
                        self.external_ssl_native_camera_encoding
                    ),
                    "action_visual_tokens": str(
                        self.external_ssl_native_action_visual_tokens
                    ),
                }
                native_mismatches = {
                    key: (saved_native.get(key), value)
                    for key, value in expected_native.items()
                    if saved_native.get(key) != value
                }
                if native_mismatches:
                    raise ValueError(
                        "Native DINO action checkpoint/model topology "
                        f"mismatch: {native_mismatches}."
                    )
        saved_teacher = payload.get("foresight_teacher")
        current_videomae_teacher = (
            bool(getattr(self, "foresight_enabled", False))
            and str(getattr(self, "foresight_teacher_type", "yak"))
            == "videomae_clip"
        )
        if saved_teacher is not None and not current_videomae_teacher:
            # LIBERO evaluation intentionally sets foresight.enabled=false:
            # projector/teacher are training-only and infer_action() never
            # reads them.  The core Ovis/Yak/Action topology remains identical,
            # so loading while dropping the projector keys is valid.
            logger.info(
                "Checkpoint contains training-only VideoMAE foresight metadata; "
                "current model has foresight disabled, so it will be ignored."
            )
        if saved_teacher is None and current_videomae_teacher:
            raise ValueError(
                "Current model expects a VideoMAE future-clip checkpoint but its "
                "teacher topology metadata is missing."
            )
        if saved_teacher is not None and current_videomae_teacher:
            # Checkpoints written before the random-VideoMAE ablation did not
            # record mode/seed; they are unambiguously pretrained seed-42 runs.
            saved_teacher = dict(saved_teacher)
            saved_teacher.setdefault("mode", "frozen_pretrained")
            saved_teacher.setdefault("random_seed", 42)
            expected_teacher = {
                "type": "videomae_clip",
                "mode": str(self.foresight_teacher_mode),
                "random_seed": int(self.foresight_teacher_random_seed),
                "clip_frames": int(self.foresight_teacher_clip_frames),
                "grid_size": int(self.foresight_teacher_grid_size),
                "hidden_size": int(self.foresight_teacher_hidden_dim),
                "student_layer": int(self.foresight_layer),
                "offsets": [int(offset) for offset in self.foresight_offsets],
                "temporal_target": str(self.foresight_teacher_temporal_target),
                "future_only": True,
            }
            mismatches = {
                key: (saved_teacher.get(key), value)
                for key, value in expected_teacher.items()
                if saved_teacher.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "VideoMAE foresight checkpoint/model topology mismatch: "
                    f"{mismatches}."
                )
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.stack == "flux2":
                from .lora import merge_lora_state_dict_to_plain, remap_plain_linear_keys_to_lora_base

                mot_state = merge_lora_state_dict_to_plain(mot_state)
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            logger.info(
                "Loaded MoT weights from checkpoint: missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning("First missing MoT keys: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("First unexpected MoT keys: %s", unexpected_keys[:20])
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            load_result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            logger.info(
                "Loaded legacy video expert weights: missing_keys=%d unexpected_keys=%d",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Loaded proprio_encoder weights from checkpoint.")
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def apply_trainable_policy(self) -> None:
        """Refine trainer's default DiT-only policy for parameter-efficient modes."""
        if bool(getattr(self, "external_ssl_enabled", False)):
            ssl_student = getattr(self.mot, "ssl_student", None)
            if ssl_student is None:
                raise RuntimeError("External SSL trainable policy is missing `mot.ssl_student`.")
            ssl_student.train()
            ssl_student.requires_grad_(True)
            # Mask tokens are learned by DINO's pretraining objective but are
            # not read by the unmasked dense-image forward used here.
            # Excluding them avoids carrying permanently zero-gradient
            # parameters in AdamW/DeepSpeed without changing model outputs.
            for name, parameter in ssl_student.named_parameters():
                if name == "mask_token" or name.endswith(".mask_token"):
                    parameter.requires_grad_(False)

            external_ssl_fusion = str(
                getattr(
                    self,
                    "external_ssl_action_fusion_type",
                    "layerwise_mot",
                )
            )
            native_dino_action = external_ssl_fusion in {
                "native_dino12_mot",
                "native_dino24_mot",
                "native_dinov2_large24_mot",
                "native_videomae12_mot",
            }
            direct_ssl_action = external_ssl_fusion in {
                "direct_static_kv",
                "direct_layerwise_kv",
            }
            if native_dino_action:
                # The native container owns only the DINO student, the
                # ActionDiTDINO expert, and (during training) the projector.
                # Yak remains available solely as part of the frozen Ovis
                # condition stack and is never executed by this policy.
                self.video_expert.eval()
                self.video_expert.requires_grad_(False)
                self.action_expert.train()
                self.action_expert.requires_grad_(True)
            elif direct_ssl_action:
                # Direct DINO->Action keeps Yak's lightweight pretrained
                # input/text/time/position adapters as a frozen conditioner,
                # and never executes its 18 transformer blocks.  Freeze the
                # whole registered video expert so no Yak parameter silently
                # consumes optimizer/ZeRO state; the DINO input adapter is
                # re-enabled below.
                self.video_expert.eval()
                self.video_expert.requires_grad_(False)
                for block in self.action_expert.double_blocks:
                    # Direct prefix K/V uses txt_mod, txt_norm1, and the
                    # txt_attn QKV/QK-norm.  Its text output projection and MLP
                    # never execute, so keep those copied checkpoint weights
                    # registered but out of the optimizer.
                    block.txt_attn.norm.query_norm.requires_grad_(False)
                    block.txt_attn.proj.requires_grad_(False)
                    block.txt_mlp.requires_grad_(False)

            # These official Ovis modules are bypassed by every SSL
            # action-only path.  Keep them registered for checkpoint
            # compatibility, but exclude them from the optimizer.
            if not native_dino_action:
                self.video_expert.transformer.img_in.requires_grad_(False)
                self.video_expert.transformer.final_layer.requires_grad_(False)
                external_img_in = getattr(
                    self.video_expert,
                    "external_ssl_img_in",
                    None,
                )
                if external_img_in is None:
                    raise RuntimeError("External SSL Yak input adapter is missing.")
                external_img_in.train()
                external_img_in.requires_grad_(True)

        if self.stack != "flux2":
            return
        video_expert = self.mot.mixtures["video"] if "video" in self.mot.mixtures else None
        action_expert = self.mot.mixtures["action"] if "action" in self.mot.mixtures else None
        if action_expert is not None:
            action_expert.train()
            action_expert.requires_grad_(True)
        if video_expert is None or not bool(getattr(video_expert, "flux2_lora_enabled", False)):
            return
        video_expert.train()
        for param in video_expert.parameters():
            param.requires_grad = False
        for name, param in video_expert.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)


# Compatibility aliases for historical configs and imports.
ImageWAM = JAMPolicy
