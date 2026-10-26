from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import modulate, rope_apply
from .ovis_u1_imports import ensure_ovis_u1_remote_code_importable
from jam_multimodal.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        gqa_implementation: str = "repeat",
        force_flash_attention: bool = False,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.gqa_implementation = str(gqa_implementation).strip().lower()
        if self.gqa_implementation not in {"repeat", "sdpa"}:
            raise ValueError(
                f"`gqa_implementation` must be 'repeat' or 'sdpa', got {gqa_implementation!r}."
            )
        self.force_flash_attention = bool(force_flash_attention)
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.num_kv_heads = int(getattr(first_expert, "num_kv_heads", first_expert.num_heads))
        self.attn_head_dim = int(first_expert.attn_head_dim)
        self.block_protocol = str(getattr(first_expert, "block_protocol", "wan22"))

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            protocol = str(getattr(expert, "block_protocol", "wan22"))
            num_kv_heads = int(getattr(expert, "num_kv_heads", expert.num_heads))
            checks = {
                "num_layers": (len(expert.blocks), self.num_layers),
                "num_heads": (int(expert.num_heads), self.num_heads),
                "num_kv_heads": (num_kv_heads, self.num_kv_heads),
                "attn_head_dim": (int(expert.attn_head_dim), self.attn_head_dim),
                "block_protocol": (protocol, self.block_protocol),
            }
            for attr, (got, expected) in checks.items():
                if got != expected:
                    raise ValueError(f"All experts must share {attr}; got {got} vs {expected} for expert {name}.")

        logger.info(
            "Initialized MoT with experts=%s protocol=%s layers=%d heads=%d kv_heads=%d head_dim=%d",
            self.expert_order,
            self.block_protocol,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.attn_head_dim,
        )
        logger.info(
            "MoT attention config: gqa_implementation=%s force_flash_attention=%s",
            self.gqa_implementation,
            self.force_flash_attention,
        )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _format_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_len, key_len):
                raise ValueError(f"2D attention mask must be {(query_len, key_len)}, got {tuple(mask.shape)}")
            return mask.view(1, 1, query_len, key_len)
        if mask.ndim == 3:
            if mask.shape[0] != batch_size or tuple(mask.shape[1:]) != (query_len, key_len):
                raise ValueError(
                    f"3D attention mask must be {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[-2:]) != (query_len, key_len):
                raise ValueError(
                    f"4D attention mask must end with {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(f"attention_mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = q_cat.shape
        key_len = k_cat.shape[1]
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.attn_head_dim
        attn_mask = self._format_attention_mask(attention_mask, batch_size, query_len, key_len, q_cat.device)

        def _sdpa_context():
            if not self.force_flash_attention:
                return nullcontext()
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
            except Exception as exc:  # pragma: no cover - depends on torch build
                raise RuntimeError("`force_flash_attention=True` requires torch.nn.attention.sdpa_kernel.") from exc
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            force_flash_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

        def _forward(q_flat: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
            q = q_flat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            enable_gqa = False
            if H_kv != H and self.gqa_implementation == "repeat":
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            elif H_kv != H:
                enable_gqa = True
            with _sdpa_context():
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
            return out.transpose(1, 2).reshape(batch_size, query_len, H * D)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(_forward, q_cat, k_cat, v_cat, use_reentrant=False)
        return _forward(q_cat, k_cat, v_cat)

    def _build_io_wan22(self, expert, block, x: torch.Tensor, freqs: torch.Tensor, t_mod: torch.Tensor) -> dict:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        return {
            "q": q,
            "k": k,
            "v": v,
            "residual_x": x,
            "post_state": {
                "gate_msa": gate_msa,
                "shift_mlp": shift_mlp,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
            },
            "use_gradient_checkpointing": bool(getattr(expert, "use_gradient_checkpointing", False)),
        }

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> dict:
        if self.block_protocol == "wan22":
            return self._build_io_wan22(expert, block, x, freqs, t_mod)
        raise ValueError(f"Unsupported block_protocol={self.block_protocol!r}")

    @staticmethod
    def _apply_post_wan22(block, residual_x: torch.Tensor, mixed_attn_out: torch.Tensor, post_state: dict, context_payload: Optional[dict]) -> torch.Tensor:
        x = block.gate(residual_x, post_state["gate_msa"], block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), post_state["shift_mlp"], post_state["scale_mlp"])
        return block.gate(x, post_state["gate_mlp"], block.ffn(mlp_input))

    def _apply_expert_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        post_state: dict,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        return self._apply_post_wan22(block, residual_x, mixed_attn_out, post_state, context_payload)

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        post_state: dict,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        def _post_fn(_mixed, _x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
            return self._apply_expert_post_block(
                block,
                _x,
                _mixed,
                {
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                },
                context_payload,
            )

        args = (
            mixed_slice,
            residual_x,
            post_state["gate_msa"],
            post_state["shift_mlp"],
            post_state["scale_mlp"],
            post_state["gate_mlp"],
        )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(_post_fn, *args, use_reentrant=False)
        return _post_fn(*args)

    def _yak_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _yak_qkv(self, attn, x: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ensure_ovis_u1_remote_code_importable()
        from ovis_u1_hf.modeling_yak import apply_rope

        batch_size, seq_len = x.shape[:2]
        qkv = attn.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.attn_head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return self._yak_flatten_heads(q), self._yak_flatten_heads(k), self._yak_flatten_heads(v)

    def _yak_single_io(self, block, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> dict:
        ensure_ovis_u1_remote_code_importable()
        from ovis_u1_hf.modeling_yak import apply_rope

        mod, _ = block.modulation(vec)
        x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)
        batch_size, seq_len = x.shape[:2]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.attn_head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = block.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._yak_flatten_heads(q),
            "k": self._yak_flatten_heads(k),
            "v": self._yak_flatten_heads(v),
            "mlp": mlp,
            "gate": mod.gate,
            "residual_x": x,
        }

    @staticmethod
    def _yak_single_post(block, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        output = block.linear2(torch.cat((mixed_attn_out, block.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output

    def _flux2_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, mod) -> dict:
        from flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def _forward_flux2(
        self,
        embeds_all: Dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, object],
    ):
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("FLUX.2 MoT requires `video` and `action` experts.")
        if not isinstance(attention_mask, dict):
            raise ValueError("FLUX.2 MoT expects attention_mask={'double_joint', 'single'}.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video embeds must be a dict with `txt` and `img` tensors.")
        txt = video_state["txt"]
        img = video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("FLUX.2 action embeds must be a tensor.")

        # Training-only foresight capture.  Layer indices are 1-based over
        # double-stream followed by single-stream blocks.  Captures sit outside
        # checkpointed attention, so the student gradient remains intact.
        capture_set = getattr(self, "_foresight_capture_layers", None)
        captured: Optional[Dict[int, torch.Tensor]] = {} if capture_set else None

        video_freqs = freqs_all["video"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        action_ids = context_all["action"]["ids"]
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=img.device, dtype=img.dtype))
        video_t_mod = t_mod_all["video"]
        action_t_mod = t_mod_all["action"]

        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = v_block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            from flux2.model import apply_rope

            q, k = apply_rope(q, k, pe_full)
            video_q = self._flux2_flatten_heads(q)
            video_k = self._flux2_flatten_heads(k)
            video_v = self._flux2_flatten_heads(v)
            action_state = a_block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            mixed = self._mixed_attention(
                torch.cat([video_q, action_state["q"]], dim=1),
                torch.cat([video_k, action_state["k"]], dim=1),
                torch.cat([video_v, action_state["v"]], dim=1),
                attention_mask["double_joint"],
            )
            video_attn, action_attn = torch.split(mixed, [txt.shape[1] + img.shape[1], action.shape[1]], dim=1)
            txt_attn, img_attn = torch.split(video_attn, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = v_block._apply_residuals(img, txt, img_attn, txt_attn, mods)
            action = a_block.apply_post(action_attn, action_state)
            if captured is not None and (layer_idx + 1) in capture_set:
                captured[layer_idx + 1] = img

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            video_state = self._flux2_video_single_io(v_block, video_stream, stream_pe, video_t_mod["single"])
            action_state = a_block.prepare_qkv(action, action_pe, action_t_mod["single"])
            mixed = self._mixed_attention(
                torch.cat([video_state["q"], action_state["q"]], dim=1),
                torch.cat([video_state["k"], action_state["k"]], dim=1),
                torch.cat([video_state["v"], action_state["v"]], dim=1),
                attention_mask["single"],
            )
            video_attn, action_attn = torch.split(mixed, [video_stream.shape[1], action.shape[1]], dim=1)
            video_stream = v_block._out(video_state["residual_x"], video_attn, video_state["mlp"], video_state["gate"])
            action = a_block.apply_post(action_attn, action_state)

            global_layer_idx = int(getattr(video_expert, "double_layers")) + layer_idx + 1
            if captured is not None and global_layer_idx in capture_set:
                captured[global_layer_idx] = video_stream[:, txt.shape[1] :]

        if captured is not None:
            self._foresight_video_hiddens = captured

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"video": {"txt": txt, "img": img}, "action": action}

    def prefill_flux2_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "flux2":
            raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        from flux2.model import apply_rope

        double_cache = []
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
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            double_cache.append({"k": flat_k, "v": flat_v})
            mixed = self._mixed_attention(flat_q, flat_k, flat_v, attention_mask["double_joint"])
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(block, video_stream, stream_pe, video_t_mod["single"])
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = block._out(state["residual_x"], mixed, state["mlp"], state["gate"])

        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_flux2_video_only(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        capture_layers: Optional[set[int]] = None,
        video_expert: Optional[torch.nn.Module] = None,
        early_exit_after_last_capture: bool = False,
    ) -> dict[str, object]:
        """Run only the FLUX.2 video stream and optionally capture layers.

        Video/text/reference tokens never attend action tokens under the
        JAMPolicy mask, so this is exactly the video side of ``_forward_flux2``.
        It is used by the frozen foresight teacher; requested layers are
        1-indexed over double+single blocks.
        """
        if self.block_protocol != "flux2":
            raise ValueError("`forward_flux2_video_only` requires block_protocol='flux2'.")
        if not isinstance(attention_mask, dict):
            raise ValueError("FLUX.2 video-only forward expects a dict attention mask.")

        expert = video_expert if video_expert is not None else self.mixtures["video"]
        capture_set = set(capture_layers) if capture_layers else set()
        captured: Dict[int, torch.Tensor] = {}
        double_layers = int(getattr(expert, "double_layers"))
        single_layers = int(getattr(expert, "single_layers"))
        total_layers = double_layers + single_layers
        invalid_layers = [
            layer
            for layer in capture_set
            if not isinstance(layer, int)
            or isinstance(layer, bool)
            or layer < 1
            or layer > total_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"`capture_layers` must contain 1-indexed integers in [1, {total_layers}], "
                f"got {sorted(invalid_layers, key=repr)}."
            )
        if early_exit_after_last_capture and not capture_set:
            raise ValueError("Early exit requires at least one requested capture layer.")

        stop_layer = max(capture_set) if early_exit_after_last_capture else total_layers
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        from flux2.model import apply_rope

        double_layers_to_run = min(double_layers, stop_layer)
        for layer_idx in range(double_layers_to_run):
            block = expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            mixed = self._mixed_attention(
                flat_q,
                flat_k,
                flat_v,
                attention_mask["double_joint"],
            )
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)
            if (layer_idx + 1) in capture_set:
                captured[layer_idx + 1] = img

        if stop_layer <= double_layers:
            return {
                "captured": captured,
                "final_video": {"txt": txt, "img": img},
                "executed_layers": stop_layer,
            }

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_layers_to_run = min(single_layers, stop_layer - double_layers)
        for layer_idx in range(single_layers_to_run):
            block = expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(
                block,
                video_stream,
                stream_pe,
                video_t_mod["single"],
            )
            mixed = self._mixed_attention(
                state["q"],
                state["k"],
                state["v"],
                attention_mask["single"],
            )
            video_stream = block._out(
                state["residual_x"],
                mixed,
                state["mlp"],
                state["gate"],
            )
            global_layer_idx = double_layers + layer_idx + 1
            if global_layer_idx in capture_set:
                captured[global_layer_idx] = video_stream[:, txt.shape[1] :]

        return {
            "captured": captured,
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
            "executed_layers": double_layers + single_layers_to_run,
        }

    def forward_flux2_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: dict[str, object],
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        if self.block_protocol != "flux2":
            raise ValueError("`forward_flux2_action_with_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=action.device, dtype=action.dtype))
        action_seq_len = int(action.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len

        def _action_mask(mask: torch.Tensor) -> torch.Tensor:
            if mask.ndim == 2:
                return mask[video_seq_len:total_seq_len, :total_seq_len]
            if mask.ndim == 3:
                return mask[:, video_seq_len:total_seq_len, :total_seq_len]
            return mask[:, :, video_seq_len:total_seq_len, :total_seq_len]

        double_mask = _action_mask(attention_mask["double_joint"])
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, double_mask)
            action = block.apply_post(mixed, state)

        single_mask = _action_mask(attention_mask["single"])
        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["single"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, single_mask)
            action = block.apply_post(mixed, state)
        return action

    def _forward_yak(
        self,
        embeds_all: Dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("Yak MoT requires `video` and `action` experts.")
        if not isinstance(attention_mask, dict):
            raise ValueError("Yak MoT expects attention_mask={'joint_double', 'image_double', 'single'}.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("Yak video embeds must be a dict with `txt` and `img` tensors.")
        txt = video_state["txt"]
        img = video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("Yak action embeds must be a tensor.")

        # Foresight (training-only): capture img-stream hiddens after the
        # requested global block indices (1-indexed over double+single).
        # Captures are inter-block activations (outside the checkpointed
        # mixed-attention), so gradients flow to them normally.
        capture_set = getattr(self, "_foresight_capture_layers", None)
        captured: Optional[Dict[int, torch.Tensor]] = {} if capture_set else None

        video_freqs = freqs_all["video"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        action_ids = context_all["action"]["ids"]
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=img.device, dtype=img.dtype))
        joint_pe = torch.cat([txt_pe, img_pe, action_pe], dim=2)
        image_pe = torch.cat([img_pe, action_pe], dim=2)
        single_pe = joint_pe

        video_vec = t_mod_all["video"]
        action_vec = t_mod_all["action"]
        double_layers = int(getattr(video_expert, "double_layers"))
        single_layers = int(getattr(video_expert, "single_layers"))

        for layer_idx in range(double_layers):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]

            img_mod1, img_mod2, img_mod3 = v_block.img_mod(video_vec)
            txt_mod1, txt_mod2 = v_block.txt_mod(video_vec)
            action_mod1, action_mod2, action_mod3 = a_block.img_mod(action_vec)

            img_norm = v_block.img_norm1(img)
            img_joint_in = (1 + img_mod1.scale) * img_norm + img_mod1.shift
            img_self_in = (1 + img_mod3.scale) * img_norm + img_mod3.shift
            txt_joint_in = (1 + txt_mod1.scale) * v_block.txt_norm1(txt) + txt_mod1.shift
            action_norm = a_block.img_norm1(action)
            action_joint_in = (1 + action_mod1.scale) * action_norm + action_mod1.shift
            action_self_in = (1 + action_mod3.scale) * action_norm + action_mod3.shift

            txt_q, txt_k, txt_v = self._yak_qkv(v_block.txt_attn, txt_joint_in, txt_pe)
            img_q, img_k, img_v = self._yak_qkv(v_block.img_attn, img_joint_in, img_pe)
            action_q, action_k, action_v = self._yak_qkv(a_block.img_attn, action_joint_in, action_pe)
            joint = self._mixed_attention(
                torch.cat([txt_q, img_q, action_q], dim=1),
                torch.cat([txt_k, img_k, action_k], dim=1),
                torch.cat([txt_v, img_v, action_v], dim=1),
                attention_mask["joint_double"],
            )
            txt_attn, img_attn, action_joint_attn = torch.split(
                joint,
                [txt.shape[1], img.shape[1], action.shape[1]],
                dim=1,
            )

            img_self_q, img_self_k, img_self_v = self._yak_qkv(v_block.img_self_attn, img_self_in, img_pe)
            action_self_q, action_self_k, action_self_v = self._yak_qkv(
                a_block.img_self_attn,
                action_self_in,
                action_pe,
            )
            image_self = self._mixed_attention(
                torch.cat([img_self_q, action_self_q], dim=1),
                torch.cat([img_self_k, action_self_k], dim=1),
                torch.cat([img_self_v, action_self_v], dim=1),
                attention_mask["image_double"],
            )
            img_self_attn, action_self_attn = torch.split(image_self, [img.shape[1], action.shape[1]], dim=1)

            img = img + img_mod1.gate * v_block.img_attn.proj(img_attn)
            img = img + img_mod3.gate * v_block.img_self_attn.proj(img_self_attn)
            img = img + img_mod2.gate * v_block.img_mlp((1 + img_mod2.scale) * v_block.img_norm2(img) + img_mod2.shift)
            txt = txt + txt_mod1.gate * v_block.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * v_block.txt_mlp((1 + txt_mod2.scale) * v_block.txt_norm2(txt) + txt_mod2.shift)

            action = action + action_mod1.gate * a_block.img_attn.proj(action_joint_attn)
            action = action + action_mod3.gate * a_block.img_self_attn.proj(action_self_attn)
            action = action + action_mod2.gate * a_block.img_mlp(
                (1 + action_mod2.scale) * a_block.img_norm2(action) + action_mod2.shift
            )

            if captured is not None and (layer_idx + 1) in capture_set:
                captured[layer_idx + 1] = img

        video_stream = torch.cat([txt, img], dim=1)
        for layer_idx in range(single_layers):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            v_state = self._yak_single_io(v_block, video_stream, video_vec, torch.cat([txt_pe, img_pe], dim=2))
            a_state = self._yak_single_io(a_block, action, action_vec, action_pe)
            mixed = self._mixed_attention(
                torch.cat([v_state["q"], a_state["q"]], dim=1),
                torch.cat([v_state["k"], a_state["k"]], dim=1),
                torch.cat([v_state["v"], a_state["v"]], dim=1),
                attention_mask["single"],
            )
            video_attn, action_attn = torch.split(mixed, [video_stream.shape[1], action.shape[1]], dim=1)
            video_stream = self._yak_single_post(v_block, video_attn, v_state)
            action = self._yak_single_post(a_block, action_attn, a_state)

            if captured is not None and (double_layers + layer_idx + 1) in capture_set:
                captured[double_layers + layer_idx + 1] = video_stream[:, txt.shape[1] :]

        if captured is not None:
            self._foresight_video_hiddens = captured

        txt_len = int(txt.shape[1])
        txt, img = video_stream[:, :txt_len], video_stream[:, txt_len:]
        return {"video": {"txt": txt, "img": img}, "action": action}

    def prefill_yak_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "yak":
            raise ValueError("`prefill_yak_video_cache` requires block_protocol='yak'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        double_cache = []
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            img_mod1, img_mod2, img_mod3 = block.img_mod(video_t_mod)
            txt_mod1, txt_mod2 = block.txt_mod(video_t_mod)
            img_norm = block.img_norm1(img)
            img_joint_in = (1 + img_mod1.scale) * img_norm + img_mod1.shift
            img_self_in = (1 + img_mod3.scale) * img_norm + img_mod3.shift
            txt_joint_in = (1 + txt_mod1.scale) * block.txt_norm1(txt) + txt_mod1.shift
            txt_q, txt_k, txt_v = self._yak_qkv(block.txt_attn, txt_joint_in, txt_pe)
            img_q, img_k, img_v = self._yak_qkv(block.img_attn, img_joint_in, img_pe)
            img_self_q, img_self_k, img_self_v = self._yak_qkv(block.img_self_attn, img_self_in, img_pe)
            double_cache.append(
                {
                    "joint_k": torch.cat([txt_k, img_k], dim=1),
                    "joint_v": torch.cat([txt_v, img_v], dim=1),
                    "image_k": img_self_k,
                    "image_v": img_self_v,
                }
            )
            joint = self._mixed_attention(
                torch.cat([txt_q, img_q], dim=1),
                torch.cat([txt_k, img_k], dim=1),
                torch.cat([txt_v, img_v], dim=1),
                attention_mask["joint_double"],
            )
            txt_attn, img_attn = torch.split(joint, [txt.shape[1], img.shape[1]], dim=1)
            img_self_attn = self._mixed_attention(img_self_q, img_self_k, img_self_v, attention_mask["image_double"])
            img = img + img_mod1.gate * block.img_attn.proj(img_attn)
            img = img + img_mod3.gate * block.img_self_attn.proj(img_self_attn)
            img = img + img_mod2.gate * block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)
            txt = txt + txt_mod1.gate * block.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._yak_single_io(block, video_stream, video_t_mod, stream_pe)
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = self._yak_single_post(block, mixed, state)
        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_yak_video_only(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        capture_layers: Optional[set[int]] = None,
        video_expert: Optional[torch.nn.Module] = None,
        early_exit_after_last_capture: bool = False,
    ) -> dict[str, object]:
        """Video-stream-only Yak forward (no action stream), same math as the
        video side of `_forward_yak` / `prefill_yak_video_cache` (video tokens
        never attend action tokens, so this equals the in-MoT computation).

        Used by the foresight teacher with a frozen `video_expert` override;
        `attention_mask` must be the video-only slices (no action rows/cols).
        `capture_layers` are 1-indexed global block indices (double+single);
        captured tensors are the img-stream hiddens after those blocks.

        When `early_exit_after_last_capture=True`, execution stops immediately
        after the deepest requested capture. This is intended for frozen
        feature teachers whose post-capture output is never consumed. The
        default stays False so generic callers retain the full-depth behavior.
        """
        if self.block_protocol != "yak":
            raise ValueError("`forward_yak_video_only` requires block_protocol='yak'.")
        expert = video_expert if video_expert is not None else self.mixtures["video"]
        capture_set = set(capture_layers) if capture_layers else set()
        captured: Dict[int, torch.Tensor] = {}
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        double_layers = int(getattr(expert, "double_layers"))
        single_layers = int(getattr(expert, "single_layers"))
        total_layers = double_layers + single_layers
        invalid_layers = [
            layer
            for layer in capture_set
            if not isinstance(layer, int)
            or isinstance(layer, bool)
            or layer < 1
            or layer > total_layers
        ]
        if invalid_layers:
            raise ValueError(
                f"`capture_layers` must contain 1-indexed integers in [1, {total_layers}], "
                f"got {sorted(invalid_layers, key=repr)}."
            )
        if early_exit_after_last_capture and not capture_set:
            raise ValueError("Early exit requires at least one requested capture layer.")

        stop_layer = max(capture_set) if early_exit_after_last_capture else total_layers
        double_layers_to_run = min(double_layers, stop_layer)
        for layer_idx in range(double_layers_to_run):
            block = expert.double_blocks[layer_idx]
            img_mod1, img_mod2, img_mod3 = block.img_mod(video_t_mod)
            txt_mod1, txt_mod2 = block.txt_mod(video_t_mod)
            img_norm = block.img_norm1(img)
            img_joint_in = (1 + img_mod1.scale) * img_norm + img_mod1.shift
            img_self_in = (1 + img_mod3.scale) * img_norm + img_mod3.shift
            txt_joint_in = (1 + txt_mod1.scale) * block.txt_norm1(txt) + txt_mod1.shift
            txt_q, txt_k, txt_v = self._yak_qkv(block.txt_attn, txt_joint_in, txt_pe)
            img_q, img_k, img_v = self._yak_qkv(block.img_attn, img_joint_in, img_pe)
            img_self_q, img_self_k, img_self_v = self._yak_qkv(block.img_self_attn, img_self_in, img_pe)
            joint = self._mixed_attention(
                torch.cat([txt_q, img_q], dim=1),
                torch.cat([txt_k, img_k], dim=1),
                torch.cat([txt_v, img_v], dim=1),
                attention_mask["joint_double"],
            )
            txt_attn, img_attn = torch.split(joint, [txt.shape[1], img.shape[1]], dim=1)
            img_self_attn = self._mixed_attention(img_self_q, img_self_k, img_self_v, attention_mask["image_double"])
            img = img + img_mod1.gate * block.img_attn.proj(img_attn)
            img = img + img_mod3.gate * block.img_self_attn.proj(img_self_attn)
            img = img + img_mod2.gate * block.img_mlp((1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift)
            txt = txt + txt_mod1.gate * block.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * block.txt_mlp((1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift)
            if (layer_idx + 1) in capture_set:
                captured[layer_idx + 1] = img

        # A requested exit inside the double-stream section does not need the
        # txt/img concatenation or any single-stream setup.
        if stop_layer <= double_layers:
            return {
                "captured": captured,
                "final_video": {"txt": txt, "img": img},
                "executed_layers": stop_layer,
            }

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_layers_to_run = min(single_layers, stop_layer - double_layers)
        for layer_idx in range(single_layers_to_run):
            block = expert.single_blocks[layer_idx]
            state = self._yak_single_io(block, video_stream, video_t_mod, stream_pe)
            mixed = self._mixed_attention(state["q"], state["k"], state["v"], attention_mask["single"])
            video_stream = self._yak_single_post(block, mixed, state)
            if (double_layers + layer_idx + 1) in capture_set:
                captured[double_layers + layer_idx + 1] = video_stream[:, txt.shape[1] :]

        return {
            "captured": captured,
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
            "executed_layers": double_layers + single_layers_to_run,
        }

    def forward_yak_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.block_protocol != "yak":
            raise ValueError("`forward_yak_action_with_video_cache` requires block_protocol='yak'.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(action_ids.to(device=action.device, dtype=action.dtype))
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            action_mod1, action_mod2, action_mod3 = block.img_mod(action_t_mod)
            action_norm = block.img_norm1(action)
            action_joint_in = (1 + action_mod1.scale) * action_norm + action_mod1.shift
            action_self_in = (1 + action_mod3.scale) * action_norm + action_mod3.shift
            action_q, action_k, action_v = self._yak_qkv(block.img_attn, action_joint_in, action_pe)
            joint = self._mixed_attention(
                action_q,
                torch.cat([cache["joint_k"], action_k], dim=1),
                torch.cat([cache["joint_v"], action_v], dim=1),
                attention_mask["joint_double"],
            )
            action_self_q, action_self_k, action_self_v = self._yak_qkv(block.img_self_attn, action_self_in, action_pe)
            image_self = self._mixed_attention(
                action_self_q,
                torch.cat([cache["image_k"], action_self_k], dim=1),
                torch.cat([cache["image_v"], action_self_v], dim=1),
                attention_mask["image_double"],
            )
            action = action + action_mod1.gate * block.img_attn.proj(joint)
            action = action + action_mod3.gate * block.img_self_attn.proj(image_self)
            action = action + action_mod2.gate * block.img_mlp(
                (1 + action_mod2.scale) * block.img_norm2(action) + action_mod2.shift
            )

        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = self._yak_single_io(block, action, action_t_mod, action_pe)
            mixed = self._mixed_attention(
                state["q"],
                torch.cat([cache["k"], state["k"]], dim=1),
                torch.cat([cache["v"], state["v"]], dim=1),
                attention_mask["single"],
            )
            action = self._yak_single_post(block, mixed, state)
        return action

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            built = self._build_expert_attention_io(expert, block, x, video_freqs, video_t_mod)
            mixed = self._mixed_attention(built["q"], built["k"], built["v"], video_attention_mask)
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=built["residual_x"],
                post_state=built["post_state"],
                use_gradient_checkpointing=built["use_gradient_checkpointing"],
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": built["k"], "v": built["v"]})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        if self.block_protocol == "yak":
            return self.forward_yak_action_with_video_cache(
                action_tokens=action_tokens,
                action_ids=action_context_payload["ids"],
                action_t_mod=action_t_mod,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
            )
        if self.block_protocol == "flux2":
            return self.forward_flux2_action_with_video_cache(
                action_tokens=action_tokens,
                action_ids=action_context_payload["ids"],
                action_t_mod=action_t_mod,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.ndim == 2:
            action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        elif attention_mask.ndim == 3:
            action_attention_mask = attention_mask[:, video_seq_len:total_seq_len, :total_seq_len]
        else:
            action_attention_mask = attention_mask[:, :, video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            built = self._build_expert_attention_io(expert, block, x, action_freqs, action_t_mod)
            k_cat = torch.cat([video_kv_cache[layer_idx]["k"], built["k"]], dim=1)
            v_cat = torch.cat([video_kv_cache[layer_idx]["v"], built["v"]], dim=1)
            mixed = self._mixed_attention(built["q"], k_cat, v_cat, action_attention_mask)
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=built["residual_x"],
                post_state=built["post_state"],
                use_gradient_checkpointing=built["use_gradient_checkpointing"],
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        if self.block_protocol == "yak":
            return self._forward_yak(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )
        if self.block_protocol == "flux2":
            return self._forward_flux2(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )
        for key in self.expert_order:
            if key not in embeds_all or key not in freqs_all or key not in t_mod_all:
                raise ValueError(f"Missing MoT inputs for expert {key!r}")

        tokens_all = {k: v for k, v in embeds_all.items()}
        for layer_idx in range(self.num_layers):
            q_chunks, k_chunks, v_chunks, seq_lens = [], [], [], []
            cached = {}
            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                built = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=tokens_all[name],
                    freqs=freqs_all[name],
                    t_mod=t_mod_all[name],
                )
                q_chunks.append(built["q"])
                k_chunks.append(built["k"])
                v_chunks.append(built["v"])
                seq_lens.append(tokens_all[name].shape[1])
                cached[name] = {"block": block, **built}

            mixed = self._mixed_attention(
                q_cat=torch.cat(q_chunks, dim=1),
                k_cat=torch.cat(k_chunks, dim=1),
                v_cat=torch.cat(v_chunks, dim=1),
                attention_mask=attention_mask,
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                item = cached[name]
                tokens_all[name] = self._apply_post_with_optional_checkpoint(
                    block=item["block"],
                    residual_x=item["residual_x"],
                    post_state=item["post_state"],
                    use_gradient_checkpointing=item["use_gradient_checkpointing"],
                    mixed_slice=mixed[:, start:end, :],
                    context_payload=context_all.get(name),
                )
                start = end
        return tokens_all
