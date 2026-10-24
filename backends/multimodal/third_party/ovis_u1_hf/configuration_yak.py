from typing import Any
from typing import Union, Optional

from transformers.configuration_utils import PretrainedConfig

__all__ = ["YakConfig"]


class YakConfig(PretrainedConfig):
    """This is the configuration class to store the configuration of an [`YakModel`].

    Args:
    """

    model_type: str = "yak"

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        vec_in_dim: int = 1536,
        context_in_dim: int = 3072,
        hidden_size: int = 1536,
        mlp_ratio: int = 4,
        num_heads: int = 12,
        depth: int = 6,
        depth_single_blocks: int = 12,
        axes_dim: list = [16, 56, 56],
        theta: int = 10_000,
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        checkpoint: bool = False,
        txt_type: str = "refiner",
        timestep_shift: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        vae_config: Optional[Union[PretrainedConfig, dict]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vec_in_dim = vec_in_dim
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.guidance_embed = guidance_embed
        self.checkpoint = checkpoint
        self.txt_type = txt_type
        self.timestep_shift = timestep_shift
        self.base_shift = base_shift
        self.max_shift = max_shift

        self.vae_config = vae_config


    