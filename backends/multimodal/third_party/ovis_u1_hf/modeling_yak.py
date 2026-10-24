from typing import Optional, Callable
import math
from dataclasses import dataclass
import collections.abc
from itertools import repeat as iter_repeat

import numpy as np
import torch
from torch import Tensor, nn
import torchvision
from torchvision import transforms
try:
    from diffusers import AutoencoderKL
except ModuleNotFoundError:  # ActionDiTYak-only environments do not need the VAE.
    AutoencoderKL = None
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from einops import rearrange, repeat

from .configuration_yak import YakConfig


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(iter_repeat(x[0], n))
            return x
        return tuple(iter_repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor=None, attn_mask=None) -> Tensor:
    if pe is None:
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "B H L D -> B L (H D)")
    else:
        q, k = apply_rope(q, k, pe)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps:float=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale_factor)
        self.eps = eps

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class TriModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.multiplier = 9
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:6]),
            ModulationOut(*out[6:]),
        )


# from https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
class DoubleStreamXBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2, img_mod3 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_cos_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_cos_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare image for self-attention
        img_self_modulated = (1 + img_mod3.scale) * img_modulated + img_mod3.shift
        img_self_qkv = attn.img_self_attn.qkv(img_self_modulated)
        img_self_q, img_self_k, img_self_v = rearrange(img_self_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_self_q, img_self_k = attn.img_self_attn.norm(img_self_q, img_self_k, img_self_v)
        txt_pe, img_pe = torch.split(pe, [txt.shape[1], img.shape[1]], dim=2)
        img_self_attn = attention(img_self_q, img_self_k, img_self_v, pe=img_pe)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod3.gate * attn.img_self_attn.proj(img_self_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
    
class DoubleStreamXBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = TriModulation(hidden_size)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_self_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamXBlockProcessor()
        self.set_processor(processor)
    
    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)

class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

    

def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")   
  
def get_activation_layer(act_type):
    """get activation layer

    Args:
        act_type (str): the activation type

    Returns:
        torch.nn.functional: the activation layer
    """
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        # Approximate `tanh` requires torch >= 1.13
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

def modulate(x, shift=None, scale=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def apply_gate(x, gate=None, tanh=False):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)

class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels, **factory_kwargs)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


















class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,    
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding_refiner(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, 
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding_refiner(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class IndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, 
        )
        self.self_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, 
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, )
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, )
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, 
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, 
        )
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, ),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = attention(q, k, v, attn_mask=attn_mask)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class CrossTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, 
        )
        self.self_attn_q = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, 
        )
        self.norm_y = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, 
        )
        self.self_attn_kv = nn.Linear(
            hidden_size, hidden_size*2, bias=qkv_bias, 
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, )
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, )
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, 
        )

        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, 
        )
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, ),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: torch.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)

        norm_x = self.norm1(x)
        q = self.self_attn_q(norm_x)
        q = rearrange(qkv, "B L (H D) -> B L H D", H=self.heads_num)
        norm_y = self.norm_y(y)
        kv = self.self_attn_kv(norm_y)
        k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)
        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)

        # Self-Attention
        attn = attention(q, k, v, attn_mask=attn_mask)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x

class IndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Module):
    """
    A single token refiner block for llm text embedding refine.
    """
    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "torch",
        enable_cls_token: bool = False,
        enable_cross_attn: bool = False,
        length: int = 29,
    ):
        super().__init__()
        self.attn_mode = attn_mode
        assert self.attn_mode == "torch", "Only support 'torch' mode for token refiner."
        self.in_channels = in_channels
        self.enable_cross_attn = enable_cross_attn
        if self.enable_cross_attn:
            self.length = length
            self.input_embedder = nn.Linear(
                in_channels//length, hidden_size, bias=True, 
            )
            self.kv_embedder = nn.Linear(
                in_channels//length*(length-1), hidden_size, bias=True, 
            )
            self.fusion = CrossTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                )
        else:
            self.input_embedder = nn.Linear(
                in_channels, hidden_size, bias=True, 
            )

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        # self.t_embedder = TimestepEmbedder(hidden_size, act_layer,)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, 
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
        )

        self.enable_cls_token = enable_cls_token
        if self.enable_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.LongTensor] = None,
    ):
        if mask is None:
            context_aware_representations = x.mean(dim=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            context_aware_representations = (x * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
        c = self.c_embedder(context_aware_representations)
        if self.enable_cross_attn:
            single_channels = self.in_channels // self.length
            x, y = torch.split(x, [single_channels, single_channels*(self.length-1)], dim=-1)
            x = self.input_embedder(x)
            y = self.kv_embedder(y)
        else:
            x = self.input_embedder(x)
        if self.enable_cls_token:
            B, L, C = x.shape
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        
        if self.enable_cross_attn:
            x = self.fusion(x, y, c)
        x = self.individual_token_refiner(x, c, mask)
        if self.enable_cls_token:
            x_global = x[:, 0]
            x = x[:, 1:]
        else:
            x_global = x.mean(dim=1)
        return dict(
            txt_fea=x,
            txt_fea_avg=x_global
        )
















__all__ = ["YakModel"]

@dataclass
class VisualGeneratorOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class YakTransformer(nn.Module):
    def __init__(self, config: YakConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(f"Got {config.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if config.guidance_embed else nn.Identity()
        )
        self.txt_type = config.txt_type
        self.txt_in = SingleTokenRefiner(
            config.context_in_dim, 
            self.hidden_size, 
            heads_num=config.num_heads * 2, 
            depth=2, 
            enable_cls_token=True
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamXBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor | None = None,
        cond_img: Tensor = None,
        cond_img_ids: Tensor = None,
    ):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img_tokens = img.shape[1]
        if cond_img is not None:
            img = torch.cat([img, cond_img], dim=1)
            img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
        img = self.img_in(img)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        txt_dict = self.txt_in(txt)
        txt = txt_dict["txt_fea"]
        y = txt_dict["txt_fea_avg"]
        vec = vec + self.vector_in(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            if self.training and self.gradient_checkpointing:
                img, txt = self._gradient_checkpointing_func(
                    block.__call__,
                    img,
                    txt,
                    vec,
                    pe,
                )
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                img = self._gradient_checkpointing_func(
                    block.__call__,
                    img,
                    vec,
                    pe,
                )
            else:
                img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if cond_img is not None:
            img = torch.split(img, img_tokens, dim=1)[0]
        return img

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_noise(
    num_samples: int,
    channel: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        channel,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

class YakPretrainedModel(PreTrainedModel):
    config_class = YakConfig
    base_model_prefix = "yak"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _supports_sdpa = True


class YakModel(YakPretrainedModel):
    def __init__(self, config: YakConfig):
        super().__init__(config)
        if AutoencoderKL is None:
            raise ImportError(
                "Constructing the full Yak image generator requires diffusers; "
                "the standalone ActionDiTYak path does not."
            )
        self.vae = AutoencoderKL.from_config(config.vae_config)
        self.backbone = YakTransformer(config)

    def get_refiner(self):
        return self.backbone.txt_in
    
    def get_cls_refiner(self):
        return self.backbone.vector_in

    def get_backbone(self):
        return self.backbone

    def get_vae(self):
        return self.vae

    def preprocess_image(self, image: Image.Image, size, convert_to_rgb=True, Norm=True, output_type="tensor"):
        image = exif_transpose(image)
        if not image.mode == "RGB" and convert_to_rgb:
            image = image.convert("RGB")

        image = torchvision.transforms.functional.resize(
            image, size, interpolation=transforms.InterpolationMode.BICUBIC
        )

        arr = np.array(image)
        h = arr.shape[0]
        w = arr.shape[1]
        crop_y = (h - size) // 2
        crop_x = (w - size) // 2
        pil_image = image.crop([crop_x, crop_y, crop_x+size, crop_y+size])
        if output_type == "pil_image":
            return pil_image
        
        image_np = arr[crop_y : crop_y + size, crop_x : crop_x + size]
        hidden_h = h // 16
        hidden_w = w // 16
        hidden_size = size // 16
        img_ids = torch.zeros(hidden_h, hidden_w, 3)
        
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(hidden_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(hidden_w)[None, :]
        crop_y = (hidden_h - hidden_size) // 2
        crop_x = (hidden_w - hidden_size) // 2
        img_ids = img_ids[crop_y : crop_y + hidden_size, crop_x : crop_x + hidden_size]
        img_ids = rearrange(img_ids, "h w c -> (h w) c")

        image_tensor = torchvision.transforms.functional.to_tensor(image_np)
        if Norm:
            image_tensor = torchvision.transforms.functional.normalize(image_tensor, 
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return pil_image, image_tensor, img_ids

    def process_image_aspectratio(self, image, size):
        w, h = image.size
        t_w, t_h = size
        resize_r = max(float(t_w)/w, float(t_h)/h)
        resize_size = (int(resize_r * h), int(resize_r * w))
        image = torchvision.transforms.functional.resize(
            image, resize_size, interpolation=transforms.InterpolationMode.BICUBIC
        )
        pil_image = torchvision.transforms.functional.center_crop(
            image, (t_h, t_w)
        )
        hidden_h = t_h // 16
        hidden_w = t_w // 16
        img_ids = torch.zeros(hidden_h, hidden_w, 3)
        
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(hidden_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(hidden_w)[None, :]
        img_ids = rearrange(img_ids, "h w c -> (h w) c")
        image_tensor = torchvision.transforms.functional.to_tensor(pil_image)
        image_tensor = torchvision.transforms.functional.normalize(image_tensor, 
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return pil_image, image_tensor, img_ids
    
    def compute_vae_encodings(self, pixel_values, with_ids=True, time=0):
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor is not None:
                model_input = model_input - self.vae.config.shift_factor
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor is not None:
                model_input = model_input * self.vae.config.scaling_factor
        # patch for transformer
        bs, c, h, w = model_input.shape
        model_input = rearrange(model_input, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if with_ids:
            img_ids = torch.zeros(h // 2, w // 2, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
            img_ids[..., 0] = time
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
            return model_input, img_ids
        else:
            return model_input

    def generate_image(
            self, 
            cond,
            height, 
            width, 
            num_steps, 
            seed, 
            no_both_cond=None, 
            no_txt_cond=None,
            img_cfg=1.0,
            txt_cfg=1.0,
            output_type="pil"
        ):
        txt = cond["txt"]
        bs = len(txt)
        channel = self.vae.config.latent_channels
        height = 16 * (height // 16)
        width = 16 * (width // 16)
        torch_device = next(self.backbone.parameters()).device
        x = get_noise(
            bs,
            channel,
            height,
            width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        # prepare inputs
        img = x
        bs, c, h, w = img.shape

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs).to(img.device)

        if "vae_pixel_values" in cond:
            img_vae_cond, cond_ids = self.compute_vae_encodings(
                pixel_values=cond["vae_pixel_values"], with_ids=True, time=1.0)
            cond_ids = cond_ids.to(img.device)

        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3).to(img.device)

        timesteps = self.get_schedule(
            num_steps, img.shape[1], shift=self.config.timestep_shift,
            base_shift=self.config.base_shift, max_shift=self.config.max_shift)
        no_both_txt = no_both_cond["txt"]
        if no_txt_cond is not None:
            no_txt_txt = no_txt_cond["txt"]
            x = self.edit_denoise(img, img_ids, 
                                  txt, txt_ids, 
                                  no_txt_txt, 
                                  no_both_txt,
                                  img_vae_cond, cond_ids.to(img.device),
                                  timesteps=timesteps, 
                                  img_cfg=img_cfg, txt_cfg=txt_cfg)
        else:
            x = self.denoise(img, img_ids, txt, txt_ids, 
                             timesteps=timesteps, cfg=txt_cfg, 
                             neg_txt=no_both_txt)
        x = unpack(x.float(), height, width)

        with torch.autocast(device_type=torch_device.type, dtype=torch.float32):
            if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor is not None:
                x = x / self.vae.config.scaling_factor
            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor is not None:
                x = x + self.vae.config.shift_factor
            x = self.vae.decode(x, return_dict=False)[0]
        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = rearrange(x, "b c h w -> b h w c")
        x = (127.5 * (x + 1.0)).cpu().byte().numpy()
        if output_type == "np":
            return x
        images = []
        for i in range(bs):
            img = Image.fromarray(x[i])
            images.append(img)
        return images


    def get_schedule(self,
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)
        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # eastimate mu based on linear estimation between two points
            mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()

    def denoise(self, 
                input_img: Tensor,
                img_ids: Tensor,
                txt: Tensor,
                txt_ids: Tensor,
                # sampling parameters
                timesteps: list[float],
                cfg: float = 1.0,
                neg_txt = None):
        bs = input_img.shape[0]
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((bs,), t_curr, dtype=input_img.dtype, device=input_img.device)
            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(txt.device)
            cond_eps = self.backbone(
                img=input_img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
            )
            txt_ids = torch.zeros(bs, neg_txt.shape[1], 3).to(neg_txt.device)
            uncond_eps = self.backbone(
                img=input_img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
            )
            pred = uncond_eps + cfg * (cond_eps - uncond_eps)
            input_img = input_img + (t_prev - t_curr) * pred
        return input_img
    
    def edit_denoise(self, 
                input_img: Tensor,
                img_ids: Tensor,
                txt: Tensor,
                txt_ids: Tensor,
                no_txt_txt: Tensor,
                no_both_txt: Tensor,
                img_cond,
                cond_img_ids,
                # sampling parameters
                timesteps: list[float],
                img_cfg: float = 1.0,
                txt_cfg: float = 1.0,):
        bs = input_img.shape[0]
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((bs * 1,), t_curr, dtype=input_img.dtype, device=input_img.device)
            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(txt.device)
            cond_eps = self.backbone(
                img=input_img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
                cond_img=img_cond,
                cond_img_ids=cond_img_ids,
            )
            txt_ids = torch.zeros(bs, no_both_txt.shape[1], 3).to(no_both_txt.device)
            no_both_eps = self.backbone(
                img=input_img,
                img_ids=img_ids,
                txt=no_both_txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
            )
            txt_ids = torch.zeros(bs, no_txt_txt.shape[1], 3).to(no_txt_txt.device)
            no_txt_eps = self.backbone(
                img=input_img,
                img_ids=img_ids,
                txt=no_txt_txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
                cond_img=img_cond,
                cond_img_ids=cond_img_ids,
            )
            pred = no_both_eps 
            pred += img_cfg * (no_txt_eps - no_both_eps) 
            pred += txt_cfg * (cond_eps - no_txt_eps)
            input_img = input_img + (t_prev - t_curr) * pred
        return input_img
