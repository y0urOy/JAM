# adapted from https://huggingface.co/apple/aimv2-huge-patch14-448 (modification: add gradient checkpoint support)
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel
from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn import flash_attn_varlen_func

from .configuration_aimv2 import AIMv2Config


__all__ = ["AIMv2Model"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


# copied from qwen2.5-vl
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
# Note: in qwen2-vl and qwen2.5-vl, 3d convolution is used.
class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.config.num_channels * self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size)
        x = self.proj(x).view(-1, self.config.hidden_size) #.flatten(2).transpose(1, 2) # token_len x hidden_size
        x = self.norm(x)
        return x

class AIMv2ViTPreprocessor(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()

        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)

        self.preserve_original_pe = config.preserve_original_pe
        self.hidden_stride = config.hidden_stride

        if self.preserve_original_pe:
            self.interpolate_pe_method = config.interpolate_pe_method
            self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor, grid_thws: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.patchifier(x)

        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(tokens)
            if self.interpolate_pe_method == 'one_dim':
                pos_embed = self.pos_embed.transpose(1,2).to(tokens.device)
            elif self.interpolate_pe_method == 'two_dim':
                ori_h = ori_w = int(self.pos_embed.shape[1] ** 0.5)
                pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0,3,1,2)
            else:
                raise TypeError("The interpolation method for pe should be one_dim, two_dim.")
            cnt = 0
            for t, h, w in grid_thws:
                num_patches = h * w
                thw = t * h * w
                if self.interpolate_pe_method == 'one_dim':
                    pe = F.interpolate(pos_embed, size=num_patches, mode='linear', align_corners=False).transpose(1,2)
                elif self.interpolate_pe_method == 'two_dim':
                    # 1, 1024, 32, 32
                    pe = F.interpolate(pos_embed, size=(h,w), mode='bicubic', align_corners=False)
                    # 1, 1024, 1024
                    pe = pe.permute(0,2,3,1).reshape(1, h*w, -1)
                # 1024, 1024
                pe = pe[0].repeat(t,1)
                # 1, 16, 2, 16, 2, 1024
                pe = pe.reshape(t, h//self.hidden_stride, self.hidden_stride, w//self.hidden_stride, self.hidden_stride, -1)
                # 1024, 1024
                pe = pe.permute(0,1,3,2,4,5).reshape(thw,-1)
                pos_embed_new[cnt:cnt+thw] = pe

                cnt += thw

            tokens = tokens + pos_embed_new
        return tokens

# copied from qwen2.5-vl
def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed

class AIMv2FlashAttention2(nn.Module):
    def __init__(self, config: AIMv2Config) -> None:
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=config.use_bias)

        self.use_rope = not config.disable_rope
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if self.use_rope:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q = q.squeeze(0)
            k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

class AIMv2Block(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.attn = AIMv2FlashAttention2(config)
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config)
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, position_embeddings: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        x = x + self.mlp(self.norm_2(x))
        return x


class AIMv2Transformer(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AIMv2Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.rotary_pos_emb = VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        
        self.hidden_stride = config.hidden_stride
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        
        self.fullatt_block_indexes = config.fullatt_block_indexes

    # copied from qwen2.5_vl
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.hidden_stride // self.patch_size # patch (after merge) number in each window

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride, # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        tokens: torch.Tensor,
        grid_thws: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        # RoPE, modified from qwen2.5_vl
        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=tokens.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = tokens.size()
        tokens = tokens.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        tokens = tokens[window_index, :, :]
        tokens = tokens.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)
        
        hidden_states = () if output_hidden_states else None
        for index, block in enumerate(self.blocks):
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                tokens = self._gradient_checkpointing_func(block.__call__, tokens, cu_seqlens_tmp, position_embeddings)
            else:
                tokens = block(tokens, cu_seqlens_tmp, position_embeddings)
            if output_hidden_states:
                tokens_ = tokens.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                hidden_states += (tokens_[reverse_indices,:].reshape(seq_len, -1),)
        tokens = self.post_trunk_norm(tokens)
        tokens = tokens.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        tokens = tokens[reverse_indices,:].reshape(seq_len, -1)
        
        return tokens, hidden_states


class AIMv2PretrainedModel(PreTrainedModel):
    config_class = AIMv2Config
    base_model_prefix = "aimv2"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2Block"]
    _supports_sdpa = True


class AIMv2Model(AIMv2PretrainedModel):
    def __init__(self, config: AIMv2Config):
        super().__init__(config)
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.use_return_dict

        x = self.preprocessor(pixel_values, grid_thws=grid_thws)
        
        x, hidden_states = self.trunk(
            x, grid_thws=grid_thws, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            res = (x,)
            res += (hidden_states,) if output_hidden_states else ()
            return res

        return BaseModelOutputWithNoAttention(
            last_hidden_state=x,
            hidden_states=hidden_states,
        )
