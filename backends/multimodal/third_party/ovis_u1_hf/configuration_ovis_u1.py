from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod

from transformers import PretrainedConfig, AutoConfig, AutoModel


def _register_config_once(model_type, config_class):
    try:
        AutoConfig.register(model_type, config_class)
    except ValueError as exc:
        if "already" not in str(exc):
            raise


def _register_model_once(config_class, model_class):
    try:
        AutoModel.register(config_class, model_class)
    except ValueError as exc:
        if "already" not in str(exc):
            raise

# Model Constants
IGNORE_ID = -100
IMAGE_TOKEN_ID = -200
VIDEO_TOKEN_ID = -201 
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
IMAGE_ATOM_ID = -300
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304]

from .configuration_aimv2 import AIMv2Config
from .modeling_aimv2 import AIMv2Model
_register_config_once("aimv2", AIMv2Config)
_register_model_once(AIMv2Config, AIMv2Model)

from .configuration_yak import YakConfig
from .modeling_yak import YakModel
_register_config_once("yak", YakConfig)
_register_model_once(YakConfig, YakModel)


# ----------------------------------------------------------------------
#                     Visual Tokenizer Configuration
# ----------------------------------------------------------------------
class BaseVisualTokenizerConfig(PretrainedConfig):
    def __init__(self,
        vocab_size=16384,
        tokenize_function="softmax",
        tau=1.0,
        depths=None,
        use_indicators=False,
        drop_cls_token=False,
        backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
        hidden_stride: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split('|')]
        self.depths = depths
        self.backbone_kwargs = {}
        self.use_indicators = use_indicators
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class Aimv2VisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "aimv2_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]

        self.image_processor_new_kwargs = {}

        if kwargs.get("min_pixels", None) is not None:
            self.image_processor_new_kwargs['min_pixels'] = kwargs.get("min_pixels")
            self.backbone_kwargs['min_pixels'] = self.min_pixels
        
        if kwargs.get("max_pixels", None) is not None:
            self.image_processor_new_kwargs['max_pixels'] = kwargs.get("max_pixels")
            self.backbone_kwargs['max_pixels'] = self.max_pixels
        
        if kwargs.get("temporal_patch_size", None) is not None:
            self.image_processor_new_kwargs['temporal_patch_size'] = kwargs.get("temporal_patch_size")
            self.backbone_kwargs['temporal_patch_size'] = self.temporal_patch_size
        
        if kwargs.get("hidden_stride", None) is not None:
            self.image_processor_new_kwargs['hidden_stride'] = kwargs.get("hidden_stride")

        if kwargs.get("patch_size", None) is not None:
            self.image_processor_new_kwargs['patch_size'] = kwargs.get("patch_size")
            self.backbone_kwargs['patch_size'] = self.patch_size

        if kwargs.get("window_size", None) is not None:
            self.backbone_kwargs['window_size'] = kwargs.get("window_size")

        if kwargs.get("hidden_stride", None) is not None:
            self.backbone_kwargs['hidden_stride'] = kwargs.get("hidden_stride")

        if kwargs.get('fullatt_block_indexes', None) is not None:
            self.backbone_kwargs['fullatt_block_indexes'] = [int(i) for i in kwargs.get('fullatt_block_indexes').replace(' ','').split('|')]
        
        if kwargs.get("preserve_original_pe", None) is not None:
            self.backbone_kwargs['preserve_original_pe'] = kwargs.get("preserve_original_pe")
        
        if kwargs.get("interpolate_pe_method", None) is not None:
            self.backbone_kwargs['interpolate_pe_method'] = kwargs.get("interpolate_pe_method")

        if kwargs.get("disable_rope", None) is not None:
            self.backbone_kwargs['disable_rope'] = kwargs.get("disable_rope")

_register_config_once("aimv2_visual_tokenizer", Aimv2VisualTokenizerConfig)



# ----------------------------------------------------------------------
#                          OvisU1 Configuration
# ----------------------------------------------------------------------
class OvisU1Config(PretrainedConfig):
    model_type = "ovis_u1"

    def __init__(self,
                 llm_config: Optional[Union[PretrainedConfig, dict]] = None,
                 visual_tokenizer_config: Optional[Union[PretrainedConfig, dict]] = None,
                 visual_generator_config: Optional[Union[PretrainedConfig, dict]] = None,
                 multimodal_max_length=2048,
                 hidden_size=None,
                 conversation_formatter_class=None,
                 llm_attn_implementation=None,
                 disable_tie_weight=False,
                 **kwargs):
        super().__init__(**kwargs)
        if llm_config is not None:
            assert isinstance(llm_config, (PretrainedConfig, dict)), \
                f"expect `llm_config` to be instance of PretrainedConfig or dict, but got {type(llm_config)} type"
            if not isinstance(llm_config, PretrainedConfig):
                model_type = llm_config['model_type']
                llm_config.pop('model_type')
                llm_config = AutoConfig.for_model(model_type, **llm_config)
        self.llm_config = llm_config
        if visual_tokenizer_config is not None:
            assert isinstance(visual_tokenizer_config, (PretrainedConfig, dict)), \
                f"expect `visual_tokenizer_config` to be instance of PretrainedConfig or dict, but got {type(visual_tokenizer_config)} type"
            if not isinstance(visual_tokenizer_config, PretrainedConfig):
                model_type = visual_tokenizer_config['model_type']
                visual_tokenizer_config.pop('model_type')
                if model_type == "aimv2_native_visual_tokenizer":
                    model_type = "aimv2_visual_tokenizer"
                if visual_tokenizer_config['backbone_config']['model_type'] == "aimv2_native":
                    visual_tokenizer_config['backbone_config']['model_type'] = "aimv2"
                visual_tokenizer_config = AutoConfig.for_model(model_type, **visual_tokenizer_config)
        self.visual_tokenizer_config = visual_tokenizer_config
        if visual_generator_config is not None:
            assert isinstance(visual_generator_config, (PretrainedConfig, dict)), \
                f"expect `visual_generator_config` to be instance of PretrainedConfig or dict, but got {type(visual_generator_config)} type"
            if not isinstance(visual_generator_config, PretrainedConfig):
                model_type = visual_generator_config['model_type']
                visual_generator_config.pop('model_type')
                visual_generator_config = AutoConfig.for_model(model_type, **visual_generator_config)
        self.visual_generator_config = visual_generator_config
        self.multimodal_max_length = multimodal_max_length
        self.hidden_size = hidden_size
        self.conversation_formatter_class = conversation_formatter_class
        self.llm_attn_implementation = llm_attn_implementation
        self.disable_tie_weight = disable_tie_weight
        

# ----------------------------------------------------------------------
#                         Conversation Formatter
# ----------------------------------------------------------------------
class ConversationFormatter(ABC):
    support_tokenizer_types = None

    def __init__(self, tokenizer):
        tokenizer_type = type(tokenizer).__name__
        assert tokenizer_type in self.support_tokenizer_types, \
            f'Invalid tokenizer type, expected one from `{self.support_tokenizer_types}`, but got `{tokenizer_type}`'
        self.tokenizer = tokenizer
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID
        self.ignore_id = IGNORE_ID
        self.im_end = None
        self.video_token = VIDEO_TOKEN
        self.video_token_id = VIDEO_TOKEN_ID

    def _tokenize_with_image_symbol(self, text):
        if text.find(self.video_token) != -1:
            token = self.video_token
            token_id = self.video_token_id
        else:
            token = self.image_token
            token_id = self.image_token_id

        text_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in
                       text.split(token)]
        token_ids = []
        num_chuck = len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            token_ids.extend(chunk)
            if i < num_chuck - 1:
                token_ids.append(token_id)
        return token_ids

    @abstractmethod
    def format(self, conversations: List[Dict], generation_preface=None):
        pass

    @abstractmethod
    def format_query(self, query, generation_preface=""):
        pass


class Qwen3ConversationFormatter(ConversationFormatter):
    support_tokenizer_types = ['QWenTokenizer', 'Qwen2TokenizerFast']

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.from2role = {
            "system": "<|im_start|>system\n",
            "human": "<|im_start|>user\n",
            "gpt": "<|im_start|>assistant\n",
            "ignored_gpt": "<|im_start|>assistant\n",
        }
        self.gpt_token_num = None
        self.im_end = "<|im_end|>\n"
        self.empty_think = "<think>\n\n</think>\n\n"

    def format(self, conversations: List[Dict], generation_preface=None, enable_thinking=False):
        if self.gpt_token_num is None:
            prefilled_think = "" if enable_thinking else self.empty_think
            self.gpt_token_num = len(
                self.tokenizer(self.from2role["gpt"] + prefilled_think, add_special_tokens=False).input_ids
            )

        if generation_preface is not None:
            conversations.append({
                "from": "gpt",
                "value": generation_preface
            })

        prompt = ""
        input_ids = []
        labels = []
        num_conversation = len(conversations)
        for i, conversation in enumerate(conversations):
            frm = conversation["from"]
            role = self.from2role[frm]
            message = conversation["value"]
            if frm == 'gpt' and not enable_thinking:
                text = role + self.empty_think + message
            else:
                text = role + message
            if i < num_conversation - 1 or generation_preface is None:
                text += self.im_end
            prompt += text
            token_ids = self._tokenize_with_image_symbol(text)
            input_ids.extend(token_ids)
            label_ids = [self.ignore_id] * len(token_ids)
            if frm == "gpt" and generation_preface is None:
                # learning `\n` following `im_end` is meaningless, so the last `\n` token is ignored in label
                label_ids[self.gpt_token_num:-1] = token_ids[self.gpt_token_num:-1]
            labels.extend(label_ids)

        assert self._tokenize_with_image_symbol(prompt) == input_ids
        assert len(input_ids) == len(labels)

        if conversations[-1]['from'] == "gpt" and generation_preface is None:
            # remove the last `\n` following `im_end` in input_ids
            input_ids.pop()
            labels.pop()

        return prompt, input_ids, labels

    def format_query(self, query, generation_preface="", enable_thinking=False):
        prompt, input_ids, _ = self.format([{
            "from": "human",
            "value": query
        }], generation_preface=generation_preface, enable_thinking=enable_thinking)

        return prompt, input_ids
