import hashlib
import os
import time
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from jam_multimodal.utils.logging_config import get_logger
from jam_multimodal.utils import misc, pytorch_utils
from jam_multimodal.utils.mem_tools import PeriodicTrim
from accelerate import PartialState
logger = get_logger(__name__)

# export IMAGEWAM_MEM_TRIM_EVERY=50          
# export IMAGEWAM_M_TRIM_THRESHOLD=65536    
# export IMAGEWAM_M_MMAP_THRESHOLD=65536   
# export IMAGEWAM_M_TOP_PAD=0
# # export MALLOC_ARENA_MAX=2


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        val_split_level: str = "episode",
        global_sample_stride=1,
        sample_index_stride: int = 1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        robotwin_camera_layout: str = "compact_288x256",
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        require_text_cache: bool = True,
        qwen_text_cache_dir: Optional[str] = None,
        qwen_context_len: int = 128,
        qwen_text_cache_format: str = "qwen2_5_vl",
        endpoint_frames_only: bool = False,
        nonidle_filter_path: Optional[str] = None,
        profile_getitem: bool = False,
        condition_frame_augmentation: Optional[dict] = None,
        video_augmentation: Optional[dict] = None,
        hetero_bridge: Optional[dict] = None,
        lerobot_meta_cache: Optional[str] = None,
        arrow_cache_dir: Optional[str] = None,
        lerobot_backend: str = "v2",
        lerobot_video_backend: Optional[str] = None,
        lerobot_v3_init_num_workers: int = 1,
        lerobot_v3_index_cache: Optional[str] = None,
        lerobot_v3_video_backend: Optional[str] = None,
        lerobot_tolerance_s: Optional[float] = None,
        episode_index_filter: Optional[dict] = None,
        slow_getitem_log_sec: float = 0.0,
        emit_uint8_video: bool = False,
    ):
        image_obs_indices = [0, num_frames - 1] if endpoint_frames_only else None
        self.slow_getitem_log_sec = float(
            os.environ.get("JAM_SLOW_GETITEM_LOG_SEC", slow_getitem_log_sec)
        )
        effective_profile_getitem = bool(profile_getitem) or self.slow_getitem_log_sec > 0.0
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            val_split_level=val_split_level,
            global_sample_stride=global_sample_stride,
            sample_index_stride=sample_index_stride,
            image_obs_indices=image_obs_indices,
            nonidle_filter_path=nonidle_filter_path,
            profile_getitem=effective_profile_getitem,
            hetero_bridge=OmegaConf.to_container(hetero_bridge, resolve=True) if isinstance(hetero_bridge, DictConfig) else hetero_bridge,
            lerobot_meta_cache=lerobot_meta_cache,
            arrow_cache_dir=arrow_cache_dir,
            lerobot_backend=lerobot_backend,
            lerobot_video_backend=lerobot_video_backend,
            lerobot_v3_init_num_workers=lerobot_v3_init_num_workers,
            lerobot_v3_index_cache=lerobot_v3_index_cache,
            lerobot_v3_video_backend=lerobot_v3_video_backend,
            lerobot_tolerance_s=lerobot_tolerance_s,
            episode_index_filter=OmegaConf.to_container(episode_index_filter, resolve=True) if isinstance(episode_index_filter, DictConfig) else episode_index_filter,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0,\
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0,\
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.robotwin_camera_layout = str(robotwin_camera_layout)
        self.override_instruction = override_instruction
        self.require_text_cache = bool(require_text_cache)
        self.qwen_text_cache_dir = qwen_text_cache_dir
        self.qwen_context_len = int(qwen_context_len)
        self.qwen_text_cache_format = str(qwen_text_cache_format)
        self.endpoint_frames_only = bool(endpoint_frames_only)
        self.profile_getitem = effective_profile_getitem
        # Qwen's official image processor immediately round-trips the
        # normalized RGB tensor through uint8.  An opt-in Qwen-only path can
        # perform that exact terminal quantization in the worker, reducing
        # pinned-memory/H2D traffic by 4x.  Other backbones retain the legacy
        # float32 [-1, 1] dataset contract by default.
        self.emit_uint8_video = bool(emit_uint8_video)
        augmentation_cfg = video_augmentation if video_augmentation is not None else condition_frame_augmentation
        if augmentation_cfg is not None and is_training_set:
            # Hydra's instantiate(..., recursive=True) may have already built nested
            # _target_ configs; do not call instantiate() on an nn.Module twice.
            if isinstance(augmentation_cfg, torch.nn.Module):
                self.video_augmentation = augmentation_cfg
            else:
                self.video_augmentation = instantiate(augmentation_cfg)
        else:
            self.video_augmentation = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        # Per-worker periodic memory reclaim. Each __getitem__ leaves behind
        # small Python allocations (HF row dicts, list comprehensions,
        # pyav-decoded buffers, ...) that pile up as free chunks in the
        # worker's glibc arena. glibc only auto-trims when the free
        # top-of-heap exceeds M_TRIM_THRESHOLD (128 KB by default), so RSS
        # tends to ratchet up between auto-trims. Setting
        # IMAGEWAM_MEM_TRIM_EVERY=N forces gc.collect() + malloc_trim(0) every
        # N samples per worker (50-200 is a reasonable starting point).
        self._mem_trim = PeriodicTrim(
            every=int(os.environ.get("JAM_MEM_TRIM_EVERY", "0")),
            do_gc=os.environ.get("JAM_MEM_TRIM_GC", "1") != "0",
        )

    def __len__(self):
        return len(self.lerobot_dataset)

    @staticmethod
    def _robotwin_camera_sizes(layout: str) -> tuple[list[int], list[int], list[int]]:
        layout = str(layout).strip().lower()
        if layout in {"compact", "compact_288x256", "288x256"}:
            return [192, 256], [96, 128], [96, 128]
        if layout in {"legacy", "legacy_384x320", "384x320"}:
            return [256, 320], [128, 160], [128, 160]
        raise ValueError(
            f"Unsupported robotwin_camera_layout={layout!r}. "
            "Expected one of: compact_288x256, legacy_384x320."
        )

    def _get(self, idx):
        sample_idx = idx
        sample = None
        profile = {} if self.profile_getitem else None
        t0 = time.perf_counter()

        def _mark(name: str):
            nonlocal t0
            if profile is None:
                return
            now = time.perf_counter()
            profile[f"robot.{name}"] = now - t0
            t0 = now

        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if profile is not None and "_profile" in sample:
                profile.update(sample["_profile"])
            _mark("lerobot_dataset_get")

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        _mark("padding_retry")

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            if not self.endpoint_frames_only:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            if not self.endpoint_frames_only:
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        if not self.endpoint_frames_only:
            image_is_pad = image_is_pad[self.video_sample_indices]
        _mark("video_select")

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.video_augmentation is not None:
            video = self.video_augmentation(video)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            top_size, left_size, right_size = self._robotwin_camera_sizes(self.robotwin_camera_layout)
            cam_top = transforms_F.resize(
                video[0],
                size=top_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=left_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=right_size,
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]
        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        if self.emit_uint8_video:
            # Preserve the operation order used by JAMQwen35's tensor-native
            # processor.  The model converts these exact integer pixels back
            # to Qwen's normalized BF16 patch representation without repeating
            # the float32 quantization on the GPU.
            quantized_video = video.detach().to(dtype=torch.float32).add(1.0)
            quantized_video.mul_(127.5).round_().clamp_(0, 255)
            video = quantized_video.to(dtype=torch.uint8)
        _mark("image_postprocess")

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        expected_video_transitions = (self.num_frames - 1) // self.action_video_freq_ratio
        if self.endpoint_frames_only:
            expected_video_transitions = max(expected_video_transitions, 1)
        else:
            expected_video_transitions = video.shape[1] - 1
        if action.shape[0] % expected_video_transitions != 0:
            raise ValueError(
                f"`action` horizon must be divisible by video transitions, got {action.shape[0]} and {expected_video_transitions}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        # Optional sample provenance.  Keep these numeric identifiers through
        # collation so each sample can be mapped back to its sample index,
        # dataset root, episode and frame.  They do not affect training losses
        # or model inference.
        if "idx" in sample:
            data["sample_index"] = torch.as_tensor(sample["idx"], dtype=torch.long).reshape(-1)[0]
        for key in ("dataset_index", "episode_index", "frame_index"):
            if key in sample:
                data[key] = torch.as_tensor(sample[key], dtype=torch.long).reshape(-1)[0]
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        if "embodiment" in sample:
            data["embodiment"] = sample["embodiment"]
        if self.require_text_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            # NOTE: to keep consistent with wan2.2's behavior
            context[~context_mask] = 0.0
            context_mask = torch.ones_like(context_mask)
            data["context"] = context
            data["context_mask"] = context_mask
        if self.qwen_text_cache_dir is not None:
            text_hidden_states, text_attention_mask = self._get_cached_qwen_context(instruction)
            data["text_hidden_states"] = text_hidden_states
            data["text_attention_mask"] = text_attention_mask
        _mark("qwen_cache")
        if profile is not None:
            data["_profile"] = profile
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Build the instruction cache first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def _get_cached_qwen_context(self, prompt: str):
        if self.qwen_text_cache_dir is None:
            raise ValueError("qwen_text_cache_dir is not set.")
        cache_dir = self.qwen_text_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        suffix_by_format = {
            "qwen2_5_vl": "qwen2_5_vl",
            "qwen3_flux2": "qwen3_flux2",
        }
        if self.qwen_text_cache_format not in suffix_by_format:
            raise ValueError(
                f"Unsupported qwen_text_cache_format={self.qwen_text_cache_format!r}; "
                "expected 'qwen2_5_vl' or 'qwen3_flux2'."
            )
        suffix = suffix_by_format[self.qwen_text_cache_format]
        cache_path = os.path.join(cache_dir, f"{hashed}.{suffix}_len{self.qwen_context_len}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing Qwen text embedding cache: {cache_path}. "
                "Build it with `python tools/jam.py text-cache <recipe>` "
                "(scripts/flux2/precompute_flux2_qwen3_embeds.py, qwen_text_cache_format='qwen3_flux2')."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["text_hidden_states"]
        context_mask = payload["text_attention_mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `text_hidden_states` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `text_attention_mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen_context_len mismatch: expected {self.qwen_context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.qwen_context_len:
            raise ValueError(
                f"Cached qwen mask length mismatch: expected {self.qwen_context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        return context, context_mask

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        elapsed = time.perf_counter() - t0
        if self.slow_getitem_log_sec > 0.0 and elapsed >= self.slow_getitem_log_sec:
            profile = data.get("_profile", {})
            slow_parts = []
            if isinstance(profile, dict):
                time_profile = []
                counter_profile = []
                for key, value in profile.items():
                    if not isinstance(value, (int, float)):
                        continue
                    row = (key, float(value))
                    if (
                        key.endswith(".calls")
                        or key.endswith(".requested_frames")
                        or key.endswith(".frame_span")
                        or key.endswith(".max_frame_index")
                        or key.endswith(".frame_span_per_call_max")
                        or key.endswith(".max_frame_index_per_call_max")
                        or key.endswith(".frames_decoded")
                        or key.endswith(".pyav_eof_fallbacks")
                    ):
                        counter_profile.append(row)
                    else:
                        time_profile.append(row)
                for key, value in sorted(time_profile, key=lambda item: item[1], reverse=True)[:12]:
                    slow_parts.append(f"{key}={value:.3f}s")
                remaining = max(0, 12 - len(slow_parts))
                for key, value in sorted(counter_profile, key=lambda item: item[1], reverse=True)[:remaining]:
                    slow_parts.append(f"{key}={value:.1f}")
            logger.warning(
                "[slow-getitem] idx=%s elapsed=%.3fs %s",
                idx,
                elapsed,
                " | ".join(slow_parts),
            )
        # Force glibc to return free pages periodically (no-op when disabled).
        self._mem_trim.tick()
        return data
