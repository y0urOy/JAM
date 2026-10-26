import logging
import math
from datetime import datetime
from importlib import import_module
from typing import List, Union, Optional, Dict

import numpy as np
import PIL.Image
import torch
from torch import Tensor
from torch.nn import init
from torch.nn.functional import softmax, gumbel_softmax, pad
from torchvision import transforms
import transformers
from transformers import AutoImageProcessor
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers import CLIPImageProcessor

from .modeling_aimv2 import AIMv2Model
from .configuration_ovis_u1 import BaseVisualTokenizerConfig, Aimv2VisualTokenizerConfig
from .configuration_ovis_u1 import OvisU1Config, ConversationFormatter
from .configuration_ovis_u1 import IGNORE_ID, IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID

# ----------------------------------------------------------------------
#                            Visual Tokenizer
# ----------------------------------------------------------------------
class BaseVisualTokenizer(PreTrainedModel):
    base_model_prefix = "backbone"
    main_input_name = None
    _image_processor_class = None
    _image_processor_kwargs = {}
    _backbone_class = None

    def __init__(self, config: BaseVisualTokenizerConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if kwargs.get('train_from_scratch'):
            # for key in self._image_processor_kwargs.keys():
            #     self._image_processor_kwargs[key] = getattr(self.config, key, self._image_processor_kwargs[key])
            image_processor = self._image_processor_class.from_pretrained(kwargs['backbone_name_or_path'],
                                                                               **self._image_processor_kwargs)

            self.backbone = self._backbone_class.from_pretrained(kwargs['backbone_name_or_path'], **self.config.backbone_kwargs)
            self.config.backbone_config = self.backbone.config

            config = image_processor.to_dict()
            if getattr(self.config, 'image_processor_new_kwargs', None) is not None:
                for key in self.config.image_processor_new_kwargs.keys():
                    config[key] = self.config.image_processor_new_kwargs[key]
            if 'patch_size' not in config:
                assert getattr(self.backbone.config, 'patch_size'), "Patch size must be set."
                config['patch_size'] = self.backbone.config.patch_size
            self.image_processor = self._image_processor_class.from_dict(config)

        else:
            self.image_processor = AutoImageProcessor.from_pretrained(kwargs['image_processor_name_or_path'])
            self.backbone = AutoModel.from_config(self.config.backbone_config)
        head_dim = self.config.vocab_size - len(IMAGE_INDICATOR_IDS)  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                self.backbone.config.hidden_size * self.config.hidden_stride * self.config.hidden_stride, head_dim,
                bias=False
            ),
            torch.nn.LayerNorm(head_dim)
        )
        assert all((self.image_processor.do_resize,
                    not getattr(self.image_processor, 'do_center_crop', False),
                    self.image_processor.do_rescale,
                    self.image_processor.do_normalize
                    )), f"image_processor `{self.image_processor}` is not supported currently"

    def get_backbone(self):
        return self.backbone

    def get_monitor_tensors(self):
        raise NotImplementedError

    def get_image_processor(self):
        return self.image_processor

    def mock_input(self):
        height, width = self.get_image_size()
        return torch.zeros(1, 3, height, width), self.construct_image_placeholders((1, 1))

    def get_head(self):
        return self.head

    def get_image_size(self):
        raise NotImplementedError

    @staticmethod
    def construct_image_placeholders(grid, data_type='image'):
        if data_type == 'image':
            image_placeholders = [IMAGE_INDICATOR_IDS[0], IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS[1]]
        elif data_type == 'video':
            image_placeholders = [IMAGE_INDICATOR_IDS[2], IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS[2]]
        else:
            raise TypeError
        
        return image_placeholders

    @staticmethod
    def _partition(img_size, grid):
        w, h = img_size
        row_height = h // grid[0]
        col_width = w // grid[1]

        partition = []
        for row in range(grid[0]):
            for col in range(grid[1]):
                left = col * col_width
                upper = row * row_height
                right = w if col == grid[1] - 1 else (col + 1) * col_width
                lower = h if row == grid[0] - 1 else (row + 1) * row_height
                partition.append((left, upper, right, lower))

        return partition

    @staticmethod
    def get_best_grid(img_size, side, max_partition, covering_threshold):

        def _covering_area(left, upper, right, lower, side):
            w = right - left
            h = lower - upper
            w, h = max(w, h), min(w, h)
            if w > side:
                h = h / w * side
                w = side
            return w * h

        img_area = img_size[0] * img_size[1]

        candidate_grids = []
        for i in range(1, max_partition + 1):
            for j in range(1, max_partition + 1):
                if i * j <= max_partition:
                    candidate_grids.append((i, j))

        all_grids = []
        good_grids = []
        for grid in candidate_grids:
            partition = BaseVisualTokenizer._partition(img_size, grid)
            covering_ratio = sum([_covering_area(*p, side) for p in partition]) / img_area
            assert covering_ratio <= 1.0
            all_grids.append((grid, covering_ratio))
            if covering_ratio > covering_threshold:
                good_grids.append((grid, covering_ratio))

        if len(good_grids) > 0:
            # pick the good partition with minimum #sub_images and break the tie using covering_ratio
            return sorted(good_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][0]
        else:
            # pick the partition with maximum covering_ratio and break the tie using #sub_images
            return sorted(all_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]

    def preprocess_image(self, image: PIL.Image.Image, max_partition=4, covering_threshold=0.9, convert_to_rgb=True):
        def _preprocess(img: PIL.Image.Image, side):
            # first resize and preprocess
            w, h = img.size
            if w == h:
                new_width = new_height = side
            elif w > h:
                new_width = side
                new_height = int(h / w * new_width)
            else:
                new_height = side
                new_width = int(w / h * new_height)
            new_size = dict(height=new_height, width=new_width)
            pixel_values = self.image_processor.preprocess(img, size=new_size, return_tensors='pt')['pixel_values']

            # then pad to square
            square_values = torch.zeros([1, 3, side, side], dtype=pixel_values.dtype, device=pixel_values.device)
            new_height, new_width = pixel_values.shape[2:]
            if new_height == new_width:
                square_values[:, :, :, :] = pixel_values
            elif new_height > new_width:
                from_index = (side - new_width) // 2
                square_values[:, :, :, from_index:from_index + new_width] = pixel_values
            else:
                from_index = (side - new_height) // 2
                square_values[:, :, from_index:from_index + new_height, :] = pixel_values

            return square_values

        if convert_to_rgb and image.mode != 'RGB':
            image = image.convert('RGB')

        sides = self.get_image_size()
        if sides[0] != sides[1]:
            raise ValueError('get_image_size() returns non-square size')
        side = sides[0]
        grid = self.get_best_grid(image.size, side, max_partition, covering_threshold)
        partition = self._partition(image.size, grid)
        crops = [image.crop(p) for p in partition]
        if len(crops) > 1:
            crops.insert(0, image)
        pixel_values = torch.cat([_preprocess(crop, side) for crop in crops], dim=0)
        image_placeholders = self.construct_image_placeholders(grid)
        return pixel_values, image_placeholders

    def get_backbone_layer(self, index):
        if 'aimv2' in self.config.model_type:
            return self.backbone.trunk.blocks[index]
        else:
            return self.backbone.vision_model.encoder.layers[index]

    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        output = self.backbone(pixel_values, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=3, this leads to a token length reduction: 729 -> 81 for siglip
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride - (sqrt_l % self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride, self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride, self.config.hidden_stride, d)
            features = features.permute(0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1, self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens

class Aimv2VisualTokenizer(BaseVisualTokenizer):
    config_class = Aimv2VisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2Block"]
    _image_processor_class = CLIPImageProcessor
    _image_processor_kwargs = dict(do_center_crop=False, crop_size={'height': -1, 'width': -1}, size={'shortest_edge':-1})
    _backbone_class = AIMv2Model
    
    # Copied from qwen2_vl
    def smart_resize(self, 
        height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
    ):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        """
        
        if height < factor or width < factor:
            print(f"height:{height} or width:{width} must be larger than factor:{factor}")
            if height < width:
                width = round(factor/height*width)
                height = factor
            else:
                height = round(factor/width*height)
                width = factor

        elif max(height, width) / min(height, width) > 200:
            print(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
            if height > width:
                height = 200 * width
            else:
                width = 200 * height

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.trunk.blocks[0].attn.qkv.weight,
            backbone_top=self.backbone.trunk.blocks[-1].attn.qkv.weight,
            head=self.head[0].weight
        )

    def get_min_image_size(self):
        min_pixels = self.image_processor.min_pixels
        max_pixels = self.image_processor.max_pixels
        height = int(min_pixels**0.5)
        width = int(min_pixels**0.5)
        patch_size = self.image_processor.patch_size
        hidden_stride = self.image_processor.hidden_stride
        height, width = self.smart_resize(height, width, patch_size * hidden_stride, min_pixels, max_pixels)
        return height, width
    
    def get_image_size(self):
        min_pixels = self.image_processor.min_pixels
        max_pixels = self.image_processor.max_pixels
        num_pixels = (min_pixels+max_pixels) / 2
        height = int(num_pixels**0.5)
        width = int(num_pixels**0.5)
        patch_size = self.image_processor.patch_size
        hidden_stride = self.image_processor.hidden_stride
        height, width = self.smart_resize(height, width, patch_size * hidden_stride, min_pixels, max_pixels)
        return height, width

    def get_token_length(self, width: int,
                            height: int, 
                            n_frames: int = 1,
                            num_images: int = 1):
        patch_size = self.image_processor.patch_size
        temporal_patch_size = self.image_processor.temporal_patch_size
        hidden_stride = self.image_processor.hidden_stride
        min_pixels = self.image_processor.min_pixels
        max_pixels = self.image_processor.max_pixels
        
        max_pixels = max_pixels // num_images
        min_pixels = min(max_pixels, min_pixels)
        
        resized_height, resized_width = height, width
        resized_height, resized_width = self.smart_resize(
                    height,
                    width,
                    factor=patch_size * hidden_stride,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
       
        if n_frames % temporal_patch_size != 0:
            n_frames = n_frames + temporal_patch_size - 1
        grid_t = n_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size // hidden_stride, resized_width // patch_size // hidden_stride

        return grid_t * grid_w * grid_h

    def mock_input(self):
        height, width = self.get_min_image_size()
        return torch.zeros(1, 3, height, width), self.construct_image_placeholders((1, 1))

    def preprocess_image(self, images: Union[PIL.Image.Image, List[PIL.Image.Image]], 
                            convert_to_rgb: Optional[bool] = True, 
                            num_images: Optional[int] = 1,
                            min_pixels: Optional[int] = None, 
                            max_pixels: Optional[int] = None,
                            multimodal_type: Optional[str] = 'single_image'):


        patch_size = self.image_processor.patch_size # 14
        temporal_patch_size = self.image_processor.temporal_patch_size # 1
        hidden_stride = self.image_processor.hidden_stride # 2
        min_pixels = min_pixels or self.image_processor.min_pixels # 200704
        max_pixels = max_pixels or self.image_processor.max_pixels # 3211264
        
        max_pixels = max_pixels // num_images
        min_pixels = min(max_pixels, min_pixels)

        if not isinstance(images, list):
            images = [images]
        if multimodal_type == 'video':
            assert len(images) >= 1
        else:
            pass
        images = [image.convert("RGB") if convert_to_rgb and image.mode != 'RGB' else image for image in images ]
        # images = [np.array(image) for image in images]
        
        width, height = images[0].size
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=patch_size * hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            new_size = dict(height=resized_height, width=resized_width)
            image_pt = self.image_processor.preprocess(image, size=new_size, return_tensors="np")['pixel_values'][0]
            
            processed_images.append(image_pt)

        patches = np.array(processed_images)
        # if data_format == ChannelDimension.LAST:
        #     patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size # 1
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size # 32, 32
        
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        # 1024, 588

        image_placeholders = self.construct_image_placeholders((1, 1), data_type='video' if multimodal_type=='video' else 'image') # [-301, -300, -302, -305]
        
        # print(flatten_patches.shape, len(images))
        return torch.tensor(flatten_patches), torch.tensor([[grid_t, grid_h, grid_w]]), image_placeholders
    
    def encode(self, pixel_values, grid_thws):
        output = self.backbone(pixel_values, grid_thws, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        # default: false
        # if self.config.drop_cls_token:
        #     features = features[:, 1:, :]
        
        # refer to qwen2.5-vl patchmerger
        seq_len, _ = features.shape
        features = features.reshape(seq_len//(self.config.hidden_stride ** 2), -1)
        
        return features

    def forward(self, pixel_values, grid_thws) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = self.tokenize(logits)
        # tokens' shape is [#Token, VocabSize-5], so padding with [#Token, 5], after
        # which, tokens' shape should become [#Token, VocabSize];
        # this is different from original aimv2 which has [BatchSize, #Token, VocabSize-5]
        token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(token_len, len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=1)
        return tokens

try:
    AutoModel.register(Aimv2VisualTokenizerConfig, Aimv2VisualTokenizer)
except ValueError as exc:
    if "already" not in str(exc):
        raise




# ----------------------------------------------------------------------
#                           Visual Generator
# ----------------------------------------------------------------------
from .configuration_yak import YakConfig
from .modeling_yak import YakModel
try:
    AutoConfig.register("yak", YakConfig)
except ValueError as exc:
    if "already" not in str(exc):
        raise
try:
    AutoModel.register(YakConfig, YakModel)
except ValueError as exc:
    if "already" not in str(exc):
        raise



# ----------------------------------------------------------------------
#                               OvisU1
# ----------------------------------------------------------------------
class VisualEmbedding(torch.nn.Embedding):
    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class OvisU1PreTrainedModel(PreTrainedModel):
    config_class = OvisU1Config
    base_model_prefix = "ovis_u1"


class OvisU1(OvisU1PreTrainedModel):
    
    def __init__(self, config: OvisU1Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        attn_kwargs = dict()
        if self.config.llm_attn_implementation:
            attn_kwargs['attn_implementation'] = self.config.llm_attn_implementation
        elif int(transformers.__version__.split('.')[0]) >= 5:
            # transformers 4.x kept the shipped llm_config (`_attn_implementation_autoset: true`)
            # on eager attention; 5.x would pick sdpa.  Keep the original numerics.
            attn_kwargs['attn_implementation'] = 'eager'
        self.llm = AutoModelForCausalLM.from_config(self.config.llm_config, **attn_kwargs)
        assert self.config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"
        # transformers>=5 also reads config.json here, whose auto_map would ask whether to run the
        # repository code; tokenizer_config.json names the tokenizer class (Qwen2Tokenizer) either way.
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path, trust_remote_code=False)
        self.visual_tokenizer = AutoModel.from_config(self.config.visual_tokenizer_config,
                                                    image_processor_name_or_path=self.config.name_or_path)
        self.visual_generator = AutoModel.from_config(self.config.visual_generator_config)
        self.vte = VisualEmbedding(self.config.visual_tokenizer_config.vocab_size, self.config.hidden_size,
                                    device=self.visual_tokenizer.device, dtype=self.visual_tokenizer.dtype)

        def _merge_modules(modules_list: tuple):
            merged_modules = []
            for modules in modules_list:
                merged_modules.extend(modules if modules else [])
            return merged_modules

        self._no_split_modules = _merge_modules((self.llm._no_split_modules, self.visual_tokenizer._no_split_modules))
        self._skip_keys_device_placement = self.llm._skip_keys_device_placement
        self._keep_in_fp32_modules = _merge_modules(
            (self.llm._keep_in_fp32_modules, self.visual_tokenizer._keep_in_fp32_modules))
        self._supports_flash_attn_2 = True
        self.is_parallelizable = all(getattr(m, 'is_parallelizable', False) for m in (self.llm, self.visual_tokenizer, self.visual_generator))
        self.supports_gradient_checkpointing = all(
            (self.llm.supports_gradient_checkpointing, self.visual_tokenizer.supports_gradient_checkpointing, self.visual_generator.supports_gradient_checkpointing))
        self._supports_sdpa = all((self.llm._supports_sdpa, self.visual_tokenizer._supports_sdpa, self.visual_generator._supports_sdpa))
        if int(transformers.__version__.split('.')[0]) >= 5:
            # transformers>=5 loading needs post_init() (all_tied_weights_keys etc.).
            self.post_init()

    def get_text_tokenizer(self):
        return self.text_tokenizer

    def get_visual_tokenizer(self):
        return self.visual_tokenizer
    
    def get_visual_generator(self):
        return self.visual_generator

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        if self.config.disable_tie_weight:
            return
        if hasattr(self, 'all_tied_weights_keys'):  # transformers>=5 tying API
            return super().tie_weights(missing_keys=missing_keys, recompute_mapping=recompute_mapping)
        self.get_llm().tie_weights()

    def get_lm_head(self):
        return self.get_llm().get_output_embeddings()

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def get_conversation_formatter(self) -> ConversationFormatter:
        if getattr(self, 'conversation_formatter', None) is None:
            self.conversation_formatter = getattr(import_module(".configuration_ovis_u1", __package__),
                                                  self.config.conversation_formatter_class)(self.text_tokenizer)
        return self.conversation_formatter

    def merge_multimodal(
            self,
            text_input_ids: torch.Tensor,
            text_attention_masks: torch.Tensor,
            text_labels: Optional[torch.Tensor],
            pixel_values: Optional[torch.Tensor],
            grid_thws: Optional[torch.Tensor],
            left_padding: bool = False
    ):
        input_device = text_input_ids.device
        visual_vocab_szie = self.get_visual_tokenizer().config.vocab_size
        visual_indicator_embeds = self.get_vte()(
            torch.tensor(
                list(range(visual_vocab_szie - 5, visual_vocab_szie)),
                dtype=torch.long,
                device=self.get_visual_tokenizer().device
            )
        ).to(device=input_device)

        if self.training:
            # When training, to be compatible with deepspeed zero, each sample has to include pixel_value tensor.
            # For text-only sample, one can simply use a full zero tensor as pixel_value, which will be ignored
            # (see below in this function); so, the gradient will not be affected.
            num_images = [x.prod() // (self.visual_tokenizer.config.hidden_stride**2) for x in grid_thws]
            
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)

            visual_embeds_ = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                        split_size_or_sections=num_images, dim=0)
            


            visual_input_ids_ = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                           split_size_or_sections=num_images, dim=0)


            visual_labels_ = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                             visual_input_ids_]

            
            visual_embeds = []
            visual_input_ids = []
            visual_labels = []
            ind = 0
            for text_input_id in text_input_ids:
                image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
                n = len(image_atom_positions)
                if n > 0:
                    visual_embeds.append(visual_embeds_[ind:ind+n])
                    visual_input_ids.append(visual_input_ids_[ind:ind+n])
                    visual_labels.append(visual_labels_[ind:ind+n])
                    ind += n
                else:
                    visual_embeds.append(visual_embeds_[ind:ind+1])
                    visual_input_ids.append(visual_input_ids_[ind:ind+1])
                    visual_labels.append(visual_labels_[ind:ind+1])
                    ind += 1
                

        else:
            # TODO: Not modified yet
            # When inference, sample can include only text with `None` pixel_value
            # num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
            num_images = [x.prod() // (self.visual_tokenizer.config.hidden_stride**2) if x is not None else 0 for x in grid_thws]
            if sum(num_images) > 0:
                visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
                try:
                    visual_embeds_ = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                        split_size_or_sections=num_images, dim=0)
                except Exception as e:
                    print(e)
                    print(pixel_values.shape, grid_thws.shape, visual_tokens.shape, num_images)
            

                visual_input_ids_ = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                            split_size_or_sections=num_images, dim=0)


                visual_labels_ = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                                visual_input_ids_]
                
                visual_embeds = []
                visual_input_ids = []
                visual_labels = []
                ind = 0
                for text_input_id in text_input_ids:
                    image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
                    n = len(image_atom_positions)
                    if n > 0:
                        visual_embeds.append(visual_embeds_[ind:ind+n])
                        visual_input_ids.append(visual_input_ids_[ind:ind+n])
                        visual_labels.append(visual_labels_[ind:ind+n])
                        ind += n
                    else:
                        visual_embeds.append(visual_embeds_[ind:ind+1])
                        visual_input_ids.append(visual_input_ids_[ind:ind+1])
                        visual_labels.append(visual_labels_[ind:ind+1])
                        ind += 1
                        
            else:
                # just placeholders
                visual_embeds = [None] * len(num_images)
                visual_input_ids = [None] * len(num_images)
                visual_labels = [None] * len(num_images)
            
        # just placeholders
        if text_labels is None:
            text_labels = torch.full(text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device)

        input_embeds = []
        attention_masks = []
        labels = []
        input_img_poss = []
        for text_input_id, text_label, text_attention_mask, visual_embed, visual_input_id, visual_label in zip(
            text_input_ids, text_labels, text_attention_masks, visual_embeds, visual_input_ids, visual_labels
        ):
            placeholder_token_mask = torch.lt(text_input_id, 0)
            text_embed = self.get_wte()(torch.masked_fill(text_input_id, placeholder_token_mask, 0))
            for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
                text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
            image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
            if len(image_atom_positions) > 0:
                input_embed_parts = []
                attention_mask_parts = []
                label_parts = []
                input_img_pos_parts = []
                prev_image_atom_position = -1
                for index, image_atom_position in enumerate(image_atom_positions):
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:image_atom_position, :])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:image_atom_position])
                    input_img_pos_parts.append(
                        torch.zeros_like(text_label[prev_image_atom_position + 1:image_atom_position])
                    )
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:image_atom_position])
                    input_embed_parts.append(visual_embed[index])
                    attention_mask_parts.append(
                        torch.ones_like(visual_label[index], dtype=torch.bool))
                    label_parts.append(visual_label[index])
                    input_img_pos_parts.append(
                        torch.ones_like(visual_label[index])
                    )
                    prev_image_atom_position = image_atom_position
                if prev_image_atom_position + 1 < text_input_id.shape[0]:
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:, :])
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:])
                    input_img_pos_parts.append(
                        torch.zeros_like(text_label[prev_image_atom_position + 1:])
                    )
                input_embed = torch.cat(input_embed_parts, dim=0)
                attention_mask = torch.cat(attention_mask_parts, dim=0)
                label = torch.cat(label_parts, dim=0)
                input_img_pos = torch.cat(input_img_pos_parts, dim=0)
            else:
                input_embed = text_embed
                attention_mask = text_attention_mask
                label = text_label
                input_img_pos = torch.zeros_like(text_label)
                if self.training:
                    # Make visual_embed & visual_indicator_embeds involved in the backward graph,
                    # to be compatible with deepspeed zero and ddp.
                    input_embed += torch.sum(visual_embed[0] * 0.0) + torch.sum(visual_indicator_embeds * 0.0)
            input_embeds.append(input_embed)
            attention_masks.append(attention_mask)
            labels.append(label)
            input_img_poss.append(input_img_pos)

        batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0, left_padding=left_padding)
        batch_attention_mask = self.pad_truncate_sequence(attention_masks, batch_first=True, padding_value=False, left_padding=left_padding)
        batch_labels = self.pad_truncate_sequence(labels, batch_first=True, padding_value=IGNORE_ID, left_padding=left_padding)
        batch_input_img_labels = self.pad_truncate_sequence(input_img_poss, batch_first=True, padding_value=0.0, left_padding=left_padding)

        return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask, batch_input_img_labels

    def pad_truncate_sequence(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0, left_padding: bool = False) -> torch.Tensor:
        if left_padding == False:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
            return pad_sequence[:,:self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in sequences],batch_first=True, padding_value=padding_value).flip(dims=[1])
            return pad_sequence[:,-self.config.multimodal_max_length:]

    def preprocess_inputs(
        self,
        text_or_conversations: Union[List[Dict], str],
        images: Optional[Union[List[PIL.Image.Image], List[List[PIL.Image.Image]]]],
        generation_preface='',
        return_labels=False,
        propagate_exception=True,
        frame_selector=None,
        multimodal_type="single_image",
        fix_sample_overall_length_navit=False,
        min_pixels=None,
        max_pixels=None,
        enable_thinking=False
    ):
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{
                "from": "human",
                "value": text_or_conversations
            }]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        else:
            raise ValueError(f'[{datetime.now()}] Invalid type of `text_or_conversations`, expected `List[Dict]` or `str`,'
                             f' but got {type(text_or_conversations)}')

        if frame_selector is not None:
            conversations, images = frame_selector(conversations=conversations,frames=images,clear_prompt=True)

        # format conversations
        prompt, raw_input_ids, raw_labels = self.get_conversation_formatter().format(
            conversations, generation_preface=generation_preface, enable_thinking=enable_thinking)

        # place image placeholders
        input_ids = []
        labels = []
        pixel_values = []
        grid_thws = []
        invalidate_label = False
        image_token_indices = [i for i, v in enumerate(raw_input_ids) if v == IMAGE_TOKEN_ID or v == VIDEO_TOKEN_ID]
        last_image_token_index = -1
        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])
            labels.extend(raw_labels[head:tail])
            try:
                # currently, do not support multiple videos
                if multimodal_type == "video":
                    image = images
                else:
                    image = images[i]
                raw_pixel_values, image_grid_thws, image_placeholders = self.visual_tokenizer.preprocess_image(
                    image, num_images=len(images) if fix_sample_overall_length_navit else 1, min_pixels=min_pixels, max_pixels=max_pixels,
                    multimodal_type=multimodal_type)
            except Exception as e:
                if propagate_exception:
                    raise e
                logging.exception(e)
                invalidate_label = True
                # raw_pixel_values, image_placeholders = self.visual_tokenizer.mock_input() # TODO
                raw_pixel_values, _ = self.visual_tokenizer.mock_input()
                mock_image = transforms.ToPILImage()(raw_pixel_values[0])
                raw_pixel_values, image_grid_thws, image_placeholders = self.visual_tokenizer.preprocess_image(
                            mock_image, min_pixels=min_pixels, max_pixels=max_pixels)
                
            input_ids.extend(image_placeholders)
            labels.extend([IGNORE_ID] * len(image_placeholders))
            pixel_values.append(raw_pixel_values)
            grid_thws.append(image_grid_thws)
        input_ids.extend(raw_input_ids[last_image_token_index + 1:])
        labels.extend(raw_labels[last_image_token_index + 1:])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_ID] * len(labels) if invalidate_label else labels, dtype=torch.long)
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        grid_thws = torch.cat(grid_thws, dim=0) if len(grid_thws) > 0 else None

        if return_labels:
            return prompt, input_ids, pixel_values, grid_thws, labels
        else:
            return prompt, input_ids, pixel_values, grid_thws

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # assert inputs.shape[0] == 1, 'Currently, only support `batch_size=1`'
        _, inputs_embeds, labels, attention_mask, input_img_labels = self.merge_multimodal(
            text_input_ids=inputs,
            text_attention_masks=kwargs.pop('attention_mask'),
            text_labels=None,
            pixel_values=kwargs.pop('pixel_values'),
            grid_thws=kwargs.pop('grid_thws'),
            left_padding=True
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        return self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def generate_condition(
            self,
            inputs: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # assert inputs.shape[0] == 1, 'Currently, only support `batch_size=1`'
        _, inputs_embeds, labels, attention_mask, input_img_labels = self.merge_multimodal(
            text_input_ids=inputs,
            text_attention_masks=kwargs.pop('attention_mask'),
            text_labels=None,
            pixel_values=kwargs.pop('pixel_values'),
            grid_thws=kwargs.pop('grid_thws'),
            left_padding=True
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        device = self.llm.device
        outputs = self.llm(inputs_embeds=inputs_embeds.to(device), 
                            labels=labels.to(device), 
                            attention_mask=attention_mask.to(device), 
                            output_hidden_states=True, 
                            **kwargs)
        semantic_cond_0 = outputs.hidden_states[-1]
        semantic_cond_1 = outputs.hidden_states[-2]
        semantic_cond = torch.cat([semantic_cond_0, semantic_cond_1], dim=-1)
        return dict(
            txt=semantic_cond
        )
    
    def generate_img(
        self,
        inputs: Optional[torch.Tensor] = None,
        cond = None,
        no_both_cond = None,
        no_txt_cond = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if cond is None:
            cond = self.generate_condition(inputs, **kwargs)
        
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        num_steps = kwargs.get('num_steps', 50)
        seed = kwargs.get('seed', 42)
        img_cfg = kwargs.pop('img_cfg', 1.5)
        txt_cfg = kwargs.pop('txt_cfg', 5)
        yak_output = self.visual_generator.generate_image(
            cond=cond, no_txt_cond=no_txt_cond, no_both_cond=no_both_cond,
            height=height, width=width, 
            num_steps=num_steps, seed=seed, 
            img_cfg=img_cfg, txt_cfg=txt_cfg,
            output_type="pil")
        return yak_output
