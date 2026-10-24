import logging
import json
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        optimizer_foreach = cfg.get("optimizer_foreach", None)
        if optimizer_foreach is not None and not isinstance(optimizer_foreach, bool):
            raise TypeError(
                "`optimizer_foreach` must be bool or null, "
                f"got {type(optimizer_foreach)}."
            )
        self.optimizer_foreach = optimizer_foreach
        optimizer_fused = cfg.get("optimizer_fused", None)
        if optimizer_fused is not None and not isinstance(optimizer_fused, bool):
            raise TypeError(
                "`optimizer_fused` must be bool or null, "
                f"got {type(optimizer_fused)}."
            )
        if optimizer_foreach is True and optimizer_fused is True:
            raise ValueError(
                "`optimizer_foreach=true` and `optimizer_fused=true` are "
                "mutually exclusive AdamW implementations."
            )
        self.optimizer_fused = optimizer_fused
        optimizer_max_group_numel = cfg.get("optimizer_max_group_numel", None)
        if optimizer_max_group_numel is not None:
            optimizer_max_group_numel = int(optimizer_max_group_numel)
            if optimizer_max_group_numel <= 0:
                raise ValueError(
                    "`optimizer_max_group_numel` must be positive or null, "
                    f"got {optimizer_max_group_numel}."
                )
        self.optimizer_max_group_numel = optimizer_max_group_numel
        projector_only_cfg = cfg.get("train_foresight_projector_only", False)
        if not isinstance(projector_only_cfg, bool):
            raise TypeError(
                "`train_foresight_projector_only` must be bool, "
                f"got {type(projector_only_cfg)}."
            )
        self.train_foresight_projector_only = projector_only_cfg
        action_only_cfg = cfg.get("train_action_expert_only", False)
        if not isinstance(action_only_cfg, bool):
            raise TypeError(
                "`train_action_expert_only` must be bool, "
                f"got {type(action_only_cfg)}."
            )
        if projector_only_cfg and action_only_cfg:
            raise ValueError(
                "`train_foresight_projector_only` and `train_action_expert_only` "
                "are mutually exclusive."
            )
        self.train_action_expert_only = action_only_cfg
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        # Optional dataloader knobs (default: PyTorch defaults)
        self.prefetch_factor = (
            int(cfg.prefetch_factor) if getattr(cfg, "prefetch_factor", None) is not None else None
        )
        self.persistent_workers = bool(getattr(cfg, "persistent_workers", False))
        dataloader_non_blocking = cfg.get("dataloader_non_blocking", False)
        if not isinstance(dataloader_non_blocking, bool):
            raise TypeError(
                "`dataloader_non_blocking` must be bool, "
                f"got {type(dataloader_non_blocking)}."
            )
        self.dataloader_non_blocking = dataloader_non_blocking
        multiprocessing_context = cfg.get("dataloader_multiprocessing_context", None)
        if multiprocessing_context is not None:
            multiprocessing_context = str(multiprocessing_context).strip().lower()
            if multiprocessing_context not in {"fork", "spawn", "forkserver"}:
                raise ValueError(
                    "Unsupported dataloader_multiprocessing_context: "
                    f"{multiprocessing_context}."
                )
        self.dataloader_multiprocessing_context = multiprocessing_context
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.gather_metrics_every_step = bool(
            getattr(cfg, "gather_metrics_every_step", True)
        )
        self.save_every = int(cfg.save_every)
        self.keep_latest_state_only = bool(getattr(cfg, "keep_latest_state_only", False))
        self.save_full_state = bool(cfg.get("save_full_state", True))
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_num_samples = max(int(getattr(cfg, "eval_num_samples", 1)), 1)
        self.rank_timer_every = int(getattr(cfg, "rank_timer_every", 0) or 0)
        self.rank_timer_sync_cuda = bool(getattr(cfg, "rank_timer_sync_cuda", True))
        self.log_cuda_memory = bool(getattr(cfg, "log_cuda_memory", False))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self._weights_resume_loaded_before_prepare = False
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            # DataLoader workers already pin CPU tensors.  Let Accelerate issue
            # their H2D copies asynchronously while the base loader fetches the
            # following batch.  Tensor values/order and optimizer math are
            # unchanged; the default remains synchronous for legacy recipes.
            dataloader_config=DataLoaderConfiguration(
                non_blocking=self.dataloader_non_blocking,
            ),
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            (
                self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
                if self.accelerator.state.deepspeed_plugin is not None
                else "no_deepspeed"
            ),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.accelerator.is_main_process:
            logger.info(
                "[mem-init] main pid=%d JAM_MEM_TRIM_EVERY=%s "
                "JAM_M_TRIM_THRESHOLD=%s JAM_M_MMAP_THRESHOLD=%s "
                "JAM_M_TOP_PAD=%s MALLOC_ARENA_MAX=%s",
                os.getpid(),
                os.environ.get("JAM_MEM_TRIM_EVERY", "<unset>"),
                os.environ.get("JAM_M_TRIM_THRESHOLD", "<unset>"),
                os.environ.get("JAM_M_MMAP_THRESHOLD", "<unset>"),
                os.environ.get("JAM_M_TOP_PAD", "<unset>"),
                os.environ.get("MALLOC_ARENA_MAX", "<unset>"),
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        self._apply_optional_projector_only_train_mode(self.model)
        self._apply_optional_action_expert_only_train_mode(self.model)
        self._load_weights_checkpoint_before_prepare()
        trainable_params = [param for param in self.model.dit.parameters() if param.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(param for param in proprio_encoder.parameters() if param.requires_grad)
        if not trainable_params:
            raise ValueError("No trainable parameters found after applying trainable policy.")
        optimizer_parameters = trainable_params
        optimizer_group_numels = [sum(parameter.numel() for parameter in trainable_params)]
        if self.optimizer_max_group_numel is not None:
            parameter_groups = []
            current_group = []
            current_numel = 0
            for parameter in trainable_params:
                parameter_numel = parameter.numel()
                if (
                    current_group
                    and current_numel + parameter_numel > self.optimizer_max_group_numel
                ):
                    parameter_groups.append({"params": current_group})
                    current_group = []
                    current_numel = 0
                current_group.append(parameter)
                current_numel += parameter_numel
            if current_group:
                parameter_groups.append({"params": current_group})
            optimizer_parameters = parameter_groups
            optimizer_group_numels = [
                sum(parameter.numel() for parameter in group["params"])
                for group in parameter_groups
            ]
        self.optimizer = torch.optim.AdamW(
            optimizer_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            foreach=self.optimizer_foreach,
            fused=self.optimizer_fused,
        )
        logger.info(
            "AdamW optimizer: foreach=%s fused=%s group_numels=%s",
            self.optimizer_foreach,
            self.optimizer_fused,
            optimizer_group_numels,
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        scheduler_total_steps_cfg = getattr(cfg, "lr_scheduler_total_steps", None)
        scheduler_total_steps = (
            total_train_steps
            if scheduler_total_steps_cfg is None
            else int(scheduler_total_steps_cfg)
        )
        if scheduler_total_steps <= 0:
            raise ValueError(
                "`lr_scheduler_total_steps` must be positive or null, "
                f"got {scheduler_total_steps_cfg}."
            )
        self.lr_scheduler_total_steps = scheduler_total_steps
        self.lr_scheduler_hold_after_end = bool(
            getattr(cfg, "lr_scheduler_hold_after_end", False)
        )
        resume_learning_rate_cfg = getattr(cfg, "resume_learning_rate", None)
        self.resume_learning_rate = (
            None
            if resume_learning_rate_cfg is None
            else float(resume_learning_rate_cfg)
        )
        if self.resume_learning_rate is not None and (
            not np.isfinite(self.resume_learning_rate)
            or self.resume_learning_rate <= 0.0
        ):
            raise ValueError(
                "`resume_learning_rate` must be a finite positive value or null, "
                f"got {resume_learning_rate_cfg}."
            )
        warmup_steps_cfg = getattr(cfg, "warmup_steps", None)
        warmup_steps = (
            int(scheduler_total_steps * 0.05)
            if warmup_steps_cfg is None
            else int(warmup_steps_cfg)
        )
        logger.info(
            "Scheduler setup: type=%s training_steps=%d scheduler_total_steps=%d "
            "warmup_steps=%d hold_after_end=%s",
            cfg.lr_scheduler_type,
            total_train_steps,
            scheduler_total_steps,
            warmup_steps,
            self.lr_scheduler_hold_after_end,
        )
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_total_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self._apply_resume_learning_rate_override()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _log_cuda_peak_memory(self):
        """Log endpoint CUDA allocator peaks without changing training behavior."""

        if not self.log_cuda_memory or not torch.cuda.is_available():
            return
        gib = float(1024**3)
        local = torch.tensor(
            [
                torch.cuda.max_memory_allocated(self.accelerator.device) / gib,
                torch.cuda.max_memory_reserved(self.accelerator.device) / gib,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        gathered = self.accelerator.gather(local).reshape(-1, 2).detach().cpu()
        if not self.accelerator.is_main_process:
            return
        allocated = gathered[:, 0]
        reserved = gathered[:, 1]
        allocated_rank = int(torch.argmax(allocated).item())
        reserved_rank = int(torch.argmax(reserved).item())
        logger.info(
            "[cuda-memory] peak_allocated=max:r%d:%.2fGiB "
            "peak_reserved=max:r%d:%.2fGiB per_rank_reserved=%s",
            allocated_rank,
            allocated[allocated_rank].item(),
            reserved_rank,
            reserved[reserved_rank].item(),
            [round(float(value), 2) for value in reserved.tolist()],
        )

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )
        if self.num_workers > 0:
            if self.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = self.prefetch_factor
            # persistent_workers avoids re-forking 24 workers every epoch boundary
            # (which is both slow and visually looks like a memory leak as RSS
            # ramps from ~0 again on each restart).
            loader_kwargs["persistent_workers"] = self.persistent_workers
            if self.dataloader_multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = self.dataloader_multiprocessing_context
        if self.accelerator.is_main_process:
            logger.info(
                "[loader] num_workers=%d batch_size=%d pin_memory=%s "
                "prefetch_factor=%s persistent_workers=%s non_blocking_h2d=%s",
                self.num_workers,
                self.batch_size,
                loader_kwargs["pin_memory"],
                loader_kwargs.get("prefetch_factor", "<default=2>"),
                loader_kwargs.get("persistent_workers", False),
                self.dataloader_non_blocking,
            )
        return DataLoader(**loader_kwargs)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _rank_timer_sync(self, enabled: bool):
        if not enabled or not self.rank_timer_sync_cuda:
            return
        if self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)

    def _gather_rank_timer_stats(self, timings: dict[str, float], device: torch.device):
        names = ("data", "forward", "backward", "optimizer", "metrics", "total")
        local = torch.tensor(
            [float(timings.get(name, 0.0)) for name in names],
            device=device,
            dtype=torch.float32,
        )
        gathered = self.accelerator.gather(local).reshape(self.accelerator.num_processes, len(names))
        return names, gathered.detach().cpu()

    @staticmethod
    def _format_rank_timer_stats(names, stats: torch.Tensor) -> str:
        parts = []
        for idx, name in enumerate(names):
            values = stats[:, idx]
            max_rank = int(torch.argmax(values).item())
            parts.append(
                f"{name}=mean:{values.mean().item():.3f}s "
                f"min:{values.min().item():.3f}s "
                f"max:r{max_rank}:{values[max_rank].item():.3f}s"
            )
        return " | ".join(parts)

    def _maybe_log_omnigen2_forward_profile(self, device: torch.device):
        model = self.accelerator.unwrap_model(self.model)
        profile_tensor = getattr(model, "_last_omnigen2_forward_profile_tensor", None)
        if profile_tensor is None:
            return

        profile_tensor = profile_tensor.detach().to(device=device, dtype=torch.float32)
        gathered = self.accelerator.gather(profile_tensor)
        num_ranks = max(int(self.accelerator.num_processes), 1)
        if gathered.numel() % num_ranks != 0:
            if self.accelerator.is_main_process:
                logger.warning(
                    "Cannot format OmniGen2 forward profile: gathered shape=%s num_ranks=%d",
                    tuple(gathered.shape),
                    num_ranks,
                )
            return
        gathered = gathered.reshape(num_ranks, -1).detach().cpu()
        if not self.accelerator.is_main_process:
            return

        phase_names = list(getattr(model, "_last_omnigen2_forward_profile", {}).get("phase_names", []))
        if not phase_names:
            phase_names = ["build_inputs", "noise_sched", "video_pre", "action_pre", "mask", "mot", "post_loss"]
        num_phases = len(phase_names)
        expected = 2 * num_phases + 10
        if gathered.shape[1] < expected:
            logger.warning("OmniGen2 forward profile is too short: shape=%s expected>=%d", tuple(gathered.shape), expected)
            return

        meta = gathered[:, 2 * num_phases :]
        should_log = bool(torch.any(meta[:, 1] > 0.5).item())
        if not should_log:
            return

        durations = gathered[:, :num_phases]
        arrivals = gathered[:, num_phases : 2 * num_phases]
        parts = []
        for idx, name in enumerate(phase_names):
            values = durations[:, idx]
            arrival_values = arrivals[:, idx]
            max_rank = int(torch.argmax(values).item())
            arrival_skew = float((arrival_values.max() - arrival_values.min()).item())
            parts.append(
                f"{name}=mean:{values.mean().item():.3f}s "
                f"min:{values.min().item():.3f}s max:r{max_rank}:{values[max_rank].item():.3f}s "
                f"arr_skew:{arrival_skew:.3f}s"
            )

        total = durations.sum(dim=1)
        total_max_rank = int(torch.argmax(total).item())
        peak = meta[:, 9]
        peak_rank = int(torch.argmax(peak).item())
        logger.info(
            "[omnigen2-forward-profile] step=%d total=mean:%.3fs min:%.3fs max:r%d:%.3fs "
            "video_seq=%.0f-%.0f action_seq=%.0f-%.0f cap=%.0f-%.0f seq=%.0f-%.0f "
            "peak=max:r%d:%.2fGiB | %s",
            self.global_step,
            total.mean().item(),
            total.min().item(),
            total_max_rank,
            total[total_max_rank].item(),
            meta[:, 3].min().item(),
            meta[:, 3].max().item(),
            meta[:, 4].min().item(),
            meta[:, 4].max().item(),
            meta[:, 5].min().item(),
            meta[:, 6].max().item(),
            meta[:, 7].min().item(),
            meta[:, 8].max().item(),
            peak_rank,
            peak[peak_rank].item(),
            " | ".join(parts),
        )

    @staticmethod
    def _looks_like_weights_checkpoint(path: Path) -> bool:
        return path.suffix.lower() in {".pt", ".pth", ".bin"} or path.is_file()

    def _load_weights_checkpoint_before_prepare(self):
        resume = self.resume
        if not resume:
            return

        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            if self._looks_like_weights_checkpoint(resume_path):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
            return
        if not self._looks_like_weights_checkpoint(resume_path):
            return

        logger.info(
            "Loading weights checkpoint before accelerator.prepare so optimizer/master weights start from checkpoint: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weights_resume_loaded_before_prepare = True
        logger.warning("Loaded weights only; optimizer/scheduler/global step will start fresh.")

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if self._weights_resume_loaded_before_prepare:
            logger.info("Weights checkpoint was loaded before accelerator.prepare; skipping post-prepare weights load.")
            return
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning(
            "Loaded weights after accelerator.prepare; optimizer/master weights may not be synchronized. "
            "Prefer loading weights-only checkpoints before prepare."
        )

    def _apply_resume_learning_rate_override(self):
        """Apply an explicit LR only after restoring a complete optimizer state.

        This is intentionally restricted to a post-schedule full-state continuation.
        Otherwise the next scheduler step could silently replace the requested LR or a
        weights-only restart could be mistaken for an exact optimizer continuation.
        """
        if self.resume_learning_rate is None:
            return

        resume_path = Path(str(self.resume)) if self.resume else None
        if resume_path is None or not resume_path.is_dir():
            raise ValueError(
                "`resume_learning_rate` requires `resume` to point to a complete "
                "training-state directory."
            )
        if not self.lr_scheduler_hold_after_end:
            raise ValueError(
                "`resume_learning_rate` requires `lr_scheduler_hold_after_end=true` "
                "so the restored scheduler cannot overwrite the override."
            )
        if self.global_step < self.lr_scheduler_total_steps:
            raise ValueError(
                "`resume_learning_rate` is only supported at or after the restored "
                "scheduler endpoint: "
                f"global_step={self.global_step}, "
                f"lr_scheduler_total_steps={self.lr_scheduler_total_steps}."
            )

        old_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = self.resume_learning_rate

        # Keep scheduler diagnostics/state_dict consistent with the intentionally
        # overridden optimizer LR. The schedule remains held and is not stepped.
        scheduler = self.scheduler
        seen = set()
        while scheduler is not None and id(scheduler) not in seen:
            seen.add(id(scheduler))
            if hasattr(scheduler, "_last_lr"):
                scheduler._last_lr = [
                    self.resume_learning_rate for _ in self.optimizer.param_groups
                ]
            scheduler = getattr(scheduler, "scheduler", None)

        new_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        if not new_lrs or any(
            not np.isclose(lr, self.resume_learning_rate, rtol=0.0, atol=1e-15)
            for lr in new_lrs
        ):
            raise RuntimeError(
                "Failed to apply `resume_learning_rate` to every optimizer group: "
                f"requested={self.resume_learning_rate}, actual={new_lrs}."
            )
        logger.info(
            "Applied post-resume learning-rate override at step=%d: old=%s new=%s "
            "(scheduler held after step=%d)",
            self.global_step,
            old_lrs,
            new_lrs,
            self.lr_scheduler_total_steps,
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)
        self._apply_optional_projector_only_train_mode(model)
        self._apply_optional_action_expert_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        apply_policy = getattr(model, "apply_trainable_policy", None)
        if callable(apply_policy):
            apply_policy()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def _apply_optional_projector_only_train_mode(self, model):
        """Freeze the policy while fitting a newly attached foresight projector.

        This is an explicit weights-warm-start mode.  It is intentionally
        trainer-level and default-off so existing FLUX/Ovis/Qwen recipes keep
        their exact trainable parameter topology.  The full policy remains in
        eval mode while the projector is in train mode; therefore dropout or
        other train/eval-sensitive policy behavior cannot drift during the
        projector calibration stage.
        """

        if not self.train_foresight_projector_only:
            return
        mot = getattr(model, "mot", None)
        projector = getattr(mot, "foresight_projector", None)
        if projector is None:
            raise ValueError(
                "`train_foresight_projector_only=true` requires a registered "
                "`model.mot.foresight_projector`."
            )

        model.dit.eval()
        model.dit.requires_grad_(False)
        projector.train()
        projector.requires_grad_(True)

        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.eval()
            proprio_encoder.requires_grad_(False)

        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any("foresight_projector" not in name for name, _ in trainable):
            raise RuntimeError(
                "Projector-only trainable policy leaked non-projector parameters: "
                f"{[name for name, _ in trainable[:20]]}"
            )
        logger.info(
            "Projector-only warm-start policy: tensors=%d numel=%d",
            len(trainable),
            sum(parameter.numel() for _, parameter in trainable),
        )

    def _apply_optional_action_expert_only_train_mode(self, model):
        """Freeze a foresight-adapted visual policy and calibrate its action expert.

        This is a default-off post-training mode.  It deliberately starts from a
        weights checkpoint with a fresh, much smaller optimizer: changing the
        trainable topology is incompatible with restoring the joint optimizer's
        ZeRO state.  The visual expert and foresight projector remain registered
        and bit-identical in the resulting full checkpoint.
        """

        if not self.train_action_expert_only:
            return

        mot = getattr(model, "mot", None)
        mixtures = getattr(mot, "mixtures", None)
        if mixtures is not None and "action" in mixtures:
            action_expert = mixtures["action"]
            proprio_module = getattr(model, "proprio_encoder", None)
            action_owner = "model.mot.mixtures['action']"
        else:
            # JAMQwen35 keeps its native ActionQwen expert and proprio
            # projection directly under ``dit`` rather than wrapping them in
            # the legacy ImageWAM ``mot.mixtures`` container.  Supporting that
            # topology here is opt-in and lets a completed foresight policy be
            # calibrated without unfreezing the Qwen visual/text backbone or
            # its training-only foresight projector.
            action_expert = getattr(model, "action_expert", None)
            proprio_module = getattr(model, "proprio_projector", None)
            action_owner = "model.action_expert"
        if action_expert is None:
            raise ValueError(
                "`train_action_expert_only=true` requires either "
                "`model.mot.mixtures['action']` or `model.action_expert`."
            )

        model.dit.eval()
        model.dit.requires_grad_(False)
        action_expert.train()
        action_expert.requires_grad_(True)

        if proprio_module is not None:
            proprio_module.train()
            proprio_module.requires_grad_(True)

        allowed_ids = {id(parameter) for parameter in action_expert.parameters()}
        if proprio_module is not None:
            allowed_ids.update(id(parameter) for parameter in proprio_module.parameters())
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        leaked = [name for name, parameter in trainable if id(parameter) not in allowed_ids]
        if not trainable or leaked:
            raise RuntimeError(
                "Action-expert-only trainable policy leaked parameters: "
                f"{leaked[:20]}"
            )
        logger.info(
            "Action-expert-only calibration policy: owner=%s tensors=%d numel=%d",
            action_owner,
            len(trainable),
            sum(parameter.numel() for _, parameter in trainable),
        )

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        text_hidden_states = sample.get("text_hidden_states", None)
        text_attention_mask = sample.get("text_attention_mask", None)
        dataset_name = sample.get("dataset_name", None)
        embodiment = sample.get("embodiment", None)
        action_is_pad = sample.get("action_is_pad", None)
        action_dim_is_pad = sample.get("action_dim_is_pad", None)
        image_is_pad = sample.get("image_is_pad", None)
        proprio_is_pad = sample.get("proprio_is_pad", None)
        proprio_dim_is_pad = sample.get("proprio_dim_is_pad", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        def _batch_mask(name: str, value, expected_ndim: int):
            if value is None:
                return None
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"`sample['{name}']` must be a torch.Tensor, got {type(value)}")
            if value.ndim == expected_ndim - 1:
                value = value.unsqueeze(0)
            if value.ndim != expected_ndim:
                raise ValueError(
                    f"`sample['{name}']` must have {expected_ndim} dims after batching, got {tuple(value.shape)}"
                )
            return value

        action_is_pad = _batch_mask("action_is_pad", action_is_pad, 2)
        action_dim_is_pad = _batch_mask("action_dim_is_pad", action_dim_is_pad, 2)
        image_is_pad = _batch_mask("image_is_pad", image_is_pad, 2)
        proprio_is_pad = _batch_mask("proprio_is_pad", proprio_is_pad, 2)
        proprio_dim_is_pad = _batch_mask("proprio_dim_is_pad", proprio_dim_is_pad, 2)

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
        if text_hidden_states is not None or text_attention_mask is not None:
            if text_hidden_states is None or text_attention_mask is None:
                raise ValueError("`text_hidden_states` and `text_attention_mask` must both exist in eval sample.")
            if text_hidden_states.ndim == 2:
                text_hidden_states = text_hidden_states.unsqueeze(0)
            if text_attention_mask.ndim == 1:
                text_attention_mask = text_attention_mask.unsqueeze(0)
            if text_hidden_states.ndim != 3 or text_attention_mask.ndim != 2:
                raise ValueError(
                    "`text_hidden_states/text_attention_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(text_hidden_states.shape)} and {tuple(text_attention_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "instruction": prompt,
            "action": action,
            "proprio": proprio,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
            "image_is_pad": image_is_pad,
            "proprio_is_pad": proprio_is_pad,
            "proprio_dim_is_pad": proprio_dim_is_pad,
            "context": context,
            "context_mask": context_mask,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action_horizon": action_horizon,
            "dataset_name": dataset_name,
            "embodiment": [embodiment] if isinstance(embodiment, str) else embodiment,
        }

    @staticmethod
    def _single_eval_dataset_name(sample):
        dataset_name = sample.get("dataset_name", None)
        if dataset_name is None:
            return None
        if isinstance(dataset_name, str):
            return dataset_name
        if isinstance(dataset_name, (list, tuple)):
            if len(dataset_name) == 0:
                return None
            return str(dataset_name[0])
        return str(dataset_name)

    def _get_eval_processor(self, sample):
        if hasattr(self.val_dataset, "lerobot_dataset"):
            return self.val_dataset.lerobot_dataset.processor
        processor_by_dataset = getattr(self.val_dataset, "processor_by_dataset", None)
        if processor_by_dataset:
            dataset_name = self._single_eval_dataset_name(sample)
            if dataset_name in processor_by_dataset:
                return processor_by_dataset[dataset_name]
        processor = getattr(self.val_dataset, "processor", None)
        if processor is not None:
            return processor
        raise AttributeError(
            "Could not find a processor on val_dataset. Expected either "
            "`val_dataset.lerobot_dataset.processor` or `val_dataset.processor`."
        )

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        model_stack = str(getattr(model, "stack", ""))
        is_omnigen2_stack = model_stack == "omnigen2"
        is_ovis_u1_stack = model_stack == "ovis_u1"
        is_flux2_stack = model_stack == "flux2"
        is_dim_stack = model_stack == "dim"
        is_image_prediction_stack = is_omnigen2_stack or is_ovis_u1_stack or is_flux2_stack or is_dim_stack
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_indices = torch.randint(
            0,
            len(self.val_dataset),
            (self.eval_num_samples,),
            generator=rng,
        ).tolist()

        local_metric_rows = []
        video_path = None
        val_video_augmentation = getattr(self.val_dataset, "video_augmentation", None)
        val_has_video_augmentation = hasattr(self.val_dataset, "video_augmentation")
        if val_has_video_augmentation:
            self.val_dataset.video_augmentation = None
        try:
            for sample_slot, eval_index in enumerate(eval_indices):
                sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

                # 1. training loss
                with self.accelerator.autocast():
                    val_loss, _ = model.training_loss(sample)
                    val_loss = val_loss.float().item()

                prompt = sample["prompt"][0]
                video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
                action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
                proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
                input_image = video0[:, 0].unsqueeze(0)
                _, num_frames, _, _ = video0.shape

                # 2. inference and video saving
                infer_kwargs = {
                    "input_image": input_image,
                    "num_frames": num_frames,
                    "action": action,
                    "action_horizon": sample["action_horizon"],
                    "proprio": proprio,
                    "text_cfg_scale": 1.0,
                    "action_cfg_scale": 1.0,
                    "num_inference_steps": self.eval_num_inference_steps,
                    "seed": 42,
                    "tiled": False,
                }
                if (is_omnigen2_stack or is_flux2_stack) and sample.get("text_hidden_states") is not None:
                    infer_kwargs["prompt"] = None
                    infer_kwargs["context"] = sample["text_hidden_states"][0]
                    infer_kwargs["context_mask"] = sample["text_attention_mask"][0]
                elif sample["context"] is not None:
                    infer_kwargs["prompt"] = None
                    infer_kwargs["context"] = sample["context"][0]
                    infer_kwargs["context_mask"] = sample["context_mask"][0]
                else:
                    infer_kwargs["prompt"] = prompt

                pred = model.infer(
                    **infer_kwargs,
                )

                pred_video = pred["video"]
                pred_action = pred.get("action", None)

                # 3. inference metrics against GT video
                pred_video_tensor = pil_frames_to_video_tensor(pred_video)
                gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
                if is_image_prediction_stack:
                    gt_video_tensor = gt_video_tensor[:, -1:, :, :]

                assert pred_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval infer prediction/GT shape mismatch: "
                    f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
                ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

                action_l1 = None
                action_l2 = None
                if action is not None and pred_action is not None:
                    if sample["proprio"] is None:
                        raise ValueError("Eval sample must contain `proprio` for action denormalization.")
                    proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)

                    processor = self._get_eval_processor(sample)

                    denorm_actions = {}
                    action_meta = processor.shape_meta["action"]
                    state_meta = processor.shape_meta["state"]
                    for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                        if not isinstance(raw_action, torch.Tensor):
                            raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                        if raw_action.ndim == 2:
                            action_btd = raw_action.unsqueeze(0)
                        elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                            action_btd = raw_action
                        else:
                            raise ValueError(
                                f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                            )
                        action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                        batch = {
                            "action": action_btd,
                            "state": proprio,
                            "embodiment": sample.get("embodiment"),
                        }
                        batch = processor.action_state_merger.backward(batch)
                        batch = processor.normalizer.backward(batch)
                        merged_batch = {
                            "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                        }
                        merged_batch = processor.action_state_merger.forward(merged_batch)
                        denorm_action = merged_batch["action"].unsqueeze(0)
                        if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                            raise ValueError(
                                f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                            )
                        denorm_actions[action_name] = denorm_action

                    pred_action_denorm = denorm_actions["pred"]
                    gt_action_denorm = denorm_actions["gt"]

                    if pred_action_denorm.shape != gt_action_denorm.shape:
                        raise ValueError(
                            "Predicted action/GT action shape mismatch after denormalization: "
                            f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                        )
                    action_diff = pred_action_denorm - gt_action_denorm
                    action_valid = torch.ones_like(action_diff, dtype=torch.bool)
                    action_is_pad = sample.get("action_is_pad")
                    if isinstance(action_is_pad, torch.Tensor):
                        if action_is_pad.ndim != 2 or action_is_pad.shape != action_diff.shape[:2]:
                            raise ValueError(
                                "`action_is_pad` shape mismatch for eval action metrics: "
                                f"got {tuple(action_is_pad.shape)} vs expected {tuple(action_diff.shape[:2])}"
                            )
                        action_valid &= ~action_is_pad.to(device=action_valid.device, dtype=torch.bool).unsqueeze(-1)
                    action_dim_is_pad = sample.get("action_dim_is_pad")
                    if isinstance(action_dim_is_pad, torch.Tensor):
                        if action_dim_is_pad.ndim != 2 or action_dim_is_pad.shape[0] != action_diff.shape[0] or action_dim_is_pad.shape[1] != action_diff.shape[2]:
                            raise ValueError(
                                "`action_dim_is_pad` shape mismatch for eval action metrics: "
                                f"got {tuple(action_dim_is_pad.shape)} vs expected ({action_diff.shape[0]}, {action_diff.shape[2]})"
                            )
                        action_valid &= ~action_dim_is_pad.to(device=action_valid.device, dtype=torch.bool).unsqueeze(1)
                    valid_count = action_valid.sum().clamp(min=1)
                    action_l1 = action_diff.abs().masked_select(action_valid).sum().div(valid_count).item()
                    action_l2 = action_diff.pow(2).masked_select(action_valid).sum().div(valid_count).item()

                # 4. VAE reconstruction metrics against GT target.
                if is_omnigen2_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_omnigen2_image_latents(gt_final)
                    vae_image = model._decode_omnigen2_image_latents(vae_latents)[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_ovis_u1_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_tokens, _ = model._encode_ovis_u1_image_tokens(gt_final, time_value=0.0)
                    vae_image = model._decode_ovis_u1_image_tokens(
                        vae_tokens,
                        height=int(gt_final.shape[-2]),
                        width=int(gt_final.shape[-1]),
                    )[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_flux2_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_tokens, _ = model._encode_flux2_image_tokens(gt_final, time_value=0.0)
                    vae_image = model._decode_flux2_image_tokens(
                        vae_tokens,
                        height=int(gt_final.shape[-2]),
                        width=int(gt_final.shape[-1]),
                    )[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                elif is_dim_stack:
                    gt_final = video0[:, -1].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_dim_image_latents(gt_final)
                    vae_image = model._decode_dim_image_latents(vae_latents)[0]
                    vae_np = ((vae_image.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
                    vae_video_tensor = pil_frames_to_video_tensor([Image.fromarray(vae_np)])
                else:
                    gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
                    vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
                    vae_recon_video = model._decode_latents(vae_latents, tiled=False)
                    vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

                assert vae_video_tensor.shape == gt_video_tensor.shape, (
                    "Eval VAE reconstruction/GT shape mismatch: "
                    f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
                )

                psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
                ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

                psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
                ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

                stitched_video_tensor = torch.cat(
                    [pred_video_tensor, vae_video_tensor, gt_video_tensor],
                    dim=2,
                ).contiguous()
                stitched_frames = []
                for t in range(stitched_video_tensor.shape[1]):
                    frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
                    stitched_frames.append(Image.fromarray(frame))

                video_name = f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}"
                if self.eval_num_samples > 1:
                    video_name += f"_sample_{sample_slot:03d}"
                current_video_path = os.path.join(self.eval_dir, f"{video_name}.mp4")
                save_mp4(stitched_frames, current_video_path, fps=8)
                if video_path is None:
                    video_path = current_video_path

                action_valid = action_l2 is not None and action_l1 is not None
                local_metric_rows.append(
                    [
                        float(val_loss),
                        float(psnr_rollout_vs_gt),
                        float(ssim_rollout_vs_gt),
                        float(psnr_rollout_vs_decode),
                        float(ssim_rollout_vs_decode),
                        float(psnr_decode_vs_gt),
                        float(ssim_decode_vs_gt),
                        float(action_l2) if action_valid else 0.0,
                        float(action_l1) if action_valid else 0.0,
                        1.0 if action_valid else 0.0,
                    ]
                )
        finally:
            if val_has_video_augmentation:
                self.val_dataset.video_augmentation = val_video_augmentation
            if was_dit_training:
                self._set_dit_only_train_mode()

        local_metrics = torch.tensor(
            local_metric_rows,
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_valid_count = gathered_metrics[:, 9].sum()
        action_l2_mean = None
        action_l1_mean = None
        if action_valid_count.item() > 0:
            action_l2_mean = (gathered_metrics[:, 7].sum() / action_valid_count).item()
            action_l1_mean = (gathered_metrics[:, 8].sum() / action_valid_count).item()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
            "num_samples": int(gathered_metrics.shape[0]),
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        world_size = int(self.accelerator.num_processes)
        distributed_microbatch_size = int(self.batch_size) * world_size
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            # ``batch_in_epoch`` is topology-dependent: b48/GA1 and b24/GA2
            # consume the same global batch per optimizer step, but the latter
            # advances two distributed microbatches.  Persist the invariant
            # sample offset so a resumed run can safely change microbatch
            # topology without replaying data.
            "samples_in_epoch": int(self.batch_in_epoch)
            * distributed_microbatch_size,
            "training_topology": {
                "batch_size_per_device": int(self.batch_size),
                "gradient_accumulation_steps": int(
                    self.gradient_accumulation_steps
                ),
                "world_size": world_size,
                "global_batch_size": int(self.batch_size)
                * int(self.gradient_accumulation_steps)
                * world_size,
            },
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _prune_old_state_checkpoints(self, keep_step_tag: str):
        if not self.keep_latest_state_only:
            return
        state_root = Path(self.state_dir)
        if not state_root.exists():
            return
        for path in state_root.iterdir():
            if not path.is_dir() or path.name == keep_step_tag:
                continue
            if not re.match(r"step[_-]\d+$", path.name):
                continue
            shutil.rmtree(path)
            logger.info("Removed old training state checkpoint: %s", path)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_full_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
                self._prune_old_state_checkpoints(keep_step_tag=step_tag)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                samples_in_epoch = payload.get("samples_in_epoch")
                if samples_in_epoch is not None:
                    samples_in_epoch = int(samples_in_epoch)
                    distributed_microbatch_size = (
                        int(self.batch_size) * int(self.accelerator.num_processes)
                    )
                    if samples_in_epoch % distributed_microbatch_size:
                        raise ValueError(
                            "Saved dataloader sample offset is not divisible by "
                            "the resumed distributed microbatch: "
                            f"samples_in_epoch={samples_in_epoch}, "
                            f"batch_size={self.batch_size}, "
                            f"world_size={self.accelerator.num_processes}."
                        )
                    self.batch_in_epoch = (
                        samples_in_epoch // distributed_microbatch_size
                    )
                else:
                    # Backward compatibility for checkpoints written before
                    # sample-invariant progress metadata was introduced.
                    self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d "
                    "sample_offset=%d saved_topology=%s",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                    payload.get("training_topology"),
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _setup_trainable_grad_audit(self):
        """Register one-shot hooks before backward so the audit also works with DeepSpeed.

        DeepSpeed may consume or clear ``Parameter.grad`` before control returns from
        ``accelerator.backward``.  Tensor hooks observe the gradients while autograd
        produces them, which makes missing-gradient detection independent of the
        optimizer wrapper.
        """

        if os.environ.get("JAM_AUDIT_TRAINABLE_GRADS", "0") != "1":
            return
        if getattr(self, "_trainable_grad_audit_handles", None):
            return

        audit_model = self.accelerator.unwrap_model(self.model)
        expected = {}
        observed = {}
        handles = []

        for name, parameter in audit_model.named_parameters():
            if not parameter.requires_grad:
                continue
            expected[name] = parameter.numel()

            def record_gradient(gradient, *, parameter_name=name):
                finite = torch.isfinite(gradient.detach()).all()
                previous = observed.get(parameter_name)
                observed[parameter_name] = (
                    finite if previous is None else torch.logical_and(previous, finite)
                )
                return gradient

            handles.append(parameter.register_hook(record_gradient))

        if not expected:
            raise RuntimeError("Trainable-gradient audit found no trainable parameters.")

        self._trainable_grad_audit_expected = expected
        self._trainable_grad_audit_observed = observed
        self._trainable_grad_audit_handles = handles
        self._trainable_grad_audit_done = False
        logger.info(
            "Registered DeepSpeed-safe trainable-gradient hooks: tensors=%d numel=%d",
            len(expected),
            sum(expected.values()),
        )

    def _finish_trainable_grad_audit(self):
        if not getattr(self, "_trainable_grad_audit_handles", None):
            return
        if bool(getattr(self, "_trainable_grad_audit_done", False)):
            return

        expected = self._trainable_grad_audit_expected
        observed = self._trainable_grad_audit_observed
        missing_grad = [name for name in expected if name not in observed]

        observed_names = list(observed)
        nonfinite_grad = []
        if observed_names:
            finite_flags = torch.stack(
                [observed[name].to(dtype=torch.bool) for name in observed_names]
            )
            finite_flags = finite_flags.detach().cpu().tolist()
            nonfinite_grad = [
                name
                for name, is_finite in zip(observed_names, finite_flags)
                if not is_finite
            ]

        if missing_grad or nonfinite_grad:
            raise RuntimeError(
                "Trainable-gradient audit failed: "
                f"missing={missing_grad[:30]} "
                f"nonfinite={nonfinite_grad[:30]}."
            )

        for handle in self._trainable_grad_audit_handles:
            handle.remove()
        self._trainable_grad_audit_handles = []
        self._trainable_grad_audit_done = True
        logger.info(
            "TRAINABLE GRAD AUDIT PASSED: tensors=%d numel=%d",
            len(expected),
            sum(expected.values()),
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self._setup_trainable_grad_audit()
        fail_on_nonfinite_loss = (
            os.environ.get("JAM_FAIL_ON_NONFINITE_LOSS", "0") == "1"
        )
        async_nonfinite_check = (
            os.environ.get("JAM_ASYNC_NONFINITE_CHECK", "0") == "1"
        )
        speed_warmup_steps = int(
            os.environ.get("JAM_SPEED_WARMUP_STEPS", "0")
        )
        if speed_warmup_steps < 0:
            raise ValueError("JAM_SPEED_WARMUP_STEPS must be non-negative.")

        def assert_finite_loss(value: torch.Tensor, description: str) -> None:
            finite = torch.isfinite(value.detach()).all()
            if (
                async_nonfinite_check
                and finite.device.type == "cuda"
                and hasattr(torch, "_assert_async")
            ):
                # Preserve fail-fast protection without forcing a host/device
                # barrier between forward and backward on every healthy step.
                torch._assert_async(finite, description)
            elif not bool(finite.item()):
                raise FloatingPointError(description)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        # IMAGEWAM_PROFILE_GPU_STEPS=a:b (measurement only): record CUDA activity over steps a..b
        # and log the GPU kernel time per step, which excludes host-side gaps.
        gpu_prof_spec = os.environ.get("JAM_PROFILE_GPU_STEPS", "")
        gpu_prof = None
        gpu_prof_a = gpu_prof_b = None
        if gpu_prof_spec:
            gpu_prof_a, gpu_prof_b = (int(v) for v in gpu_prof_spec.split(":"))
        # IMAGEWAM_PROFILE_MARKERS=a:b (measurement only): CPU+CUDA profile over steps a..b, reporting the
        # inclusive GPU time per step of the optimizer update marker (Optimizer.step#...), which DeepSpeed
        # runs inside accelerator.backward, so it cannot be skipped from the trainer.
        mk_prof_spec = os.environ.get("JAM_PROFILE_MARKERS", "")
        mk_prof = None
        mk_prof_a = mk_prof_b = None
        if mk_prof_spec:
            mk_prof_a, mk_prof_b = (int(v) for v in mk_prof_spec.split(":"))
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        speed_reset_step = self.run_start_step + speed_warmup_steps

        while self.global_step < self.max_steps:
            if gpu_prof_a is not None and gpu_prof is None and self.global_step == gpu_prof_a:
                torch.cuda.synchronize()
                gpu_prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                gpu_prof.__enter__()
            if mk_prof_a is not None and mk_prof is None and self.global_step == mk_prof_a:
                torch.cuda.synchronize()
                mk_prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                mk_prof.__enter__()
            timer_active = (
                self.rank_timer_every > 0
                and (self.global_step + 1) % self.rank_timer_every == 0
            )
            step_timings = {}
            self._rank_timer_sync(timer_active)
            step_start = time.perf_counter()
            data_start = step_start
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            step_timings["data"] = time.perf_counter() - data_start

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                iter_training_losses = getattr(train_model, "iter_training_losses", None)
                loss = None
                loss_dict = {}
                forward_elapsed = 0.0
                backward_elapsed = 0.0

                if callable(iter_training_losses):
                    objective_iter = iter(iter_training_losses(sample))
                    objective_count = 0
                    while True:
                        objective_forward_start = time.perf_counter()
                        try:
                            with self.accelerator.autocast():
                                objective_loss, objective_loss_dict = next(objective_iter)
                        except StopIteration:
                            break
                        forward_elapsed += time.perf_counter() - objective_forward_start
                        objective_count += 1
                        if fail_on_nonfinite_loss:
                            assert_finite_loss(
                                objective_loss,
                                "Non-finite training objective before backward: "
                                f"rank={self.accelerator.process_index} "
                                f"step={self.global_step + 1} "
                                f"objective={objective_count}.",
                            )
                        loss = (
                            objective_loss.detach().float()
                            if loss is None
                            else loss + objective_loss.detach().float()
                        )
                        for key, value in objective_loss_dict.items():
                            loss_dict[key] = float(value)

                        objective_backward_start = time.perf_counter()
                        self.accelerator.backward(objective_loss)
                        backward_elapsed += time.perf_counter() - objective_backward_start

                    if objective_count <= 0:
                        raise RuntimeError("`iter_training_losses` yielded no training losses.")
                else:
                    forward_start = time.perf_counter()
                    with self.accelerator.autocast():
                        loss, loss_dict = train_model.training_loss(sample)
                    forward_elapsed = time.perf_counter() - forward_start
                    if fail_on_nonfinite_loss:
                        assert_finite_loss(
                            loss,
                            "Non-finite training loss before backward: "
                            f"rank={self.accelerator.process_index} "
                            f"step={self.global_step + 1}.",
                        )

                    backward_start = time.perf_counter()
                    self.accelerator.backward(loss)
                    backward_elapsed = time.perf_counter() - backward_start

                self._rank_timer_sync(timer_active)
                step_timings["forward"] = forward_elapsed
                self._maybe_log_omnigen2_forward_profile(loss.device)
                step_timings["backward"] = backward_elapsed

                if self.accelerator.sync_gradients:
                    self._finish_trainable_grad_audit()
                    optimizer_start = time.perf_counter()
                    # IMAGEWAM_SKIP_OPTIMIZER_STEP=1 (measurement only): no clipping and no update, so the
                    # GPU time of the optimizer stage can be read as a difference to a normal run.
                    if os.environ.get("JAM_SKIP_OPTIMIZER_STEP") == "1":
                        grad_norm = torch.zeros((), device=self.accelerator.device)
                    else:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    if os.environ.get("JAM_SKIP_OPTIMIZER_STEP") != "1" and not self.accelerator.optimizer_step_was_skipped:
                        # A full-state continuation can deliberately train past
                        # the endpoint of its original cosine schedule.  PyTorch's
                        # CosineAnnealingLR rises again when stepped beyond T_max,
                        # which is not the intended continuation semantics.  The
                        # opt-in hold keeps the restored endpoint LR and scheduler
                        # state unchanged, so later full-state checkpoints remain
                        # resumable with the same policy.
                        if not (
                            self.lr_scheduler_hold_after_end
                            and self.global_step >= self.lr_scheduler_total_steps
                        ):
                            self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._rank_timer_sync(timer_active)
                    step_timings["optimizer"] = time.perf_counter() - optimizer_start

                    self.global_step += 1
                    if gpu_prof is not None and self.global_step == gpu_prof_b:
                        torch.cuda.synchronize()
                        gpu_prof.__exit__(None, None, None)
                        gpu_total_us = sum(
                            (getattr(e, "self_device_time_total", 0) or 0) for e in gpu_prof.key_averages()
                        )
                        logger.info(
                            "[gpuprof] steps %d..%d: GPU time %.1f ms per step",
                            gpu_prof_a, gpu_prof_b, gpu_total_us / 1000.0 / max(1, gpu_prof_b - gpu_prof_a),
                        )
                        gpu_prof = None
                    if mk_prof is not None and self.global_step == mk_prof_b:
                        torch.cuda.synchronize()
                        mk_prof.__exit__(None, None, None)
                        nsteps = max(1, mk_prof_b - mk_prof_a)
                        totals = {}
                        for ev in mk_prof.events():
                            name = getattr(ev, "name", "") or ""
                            if name.startswith("Optimizer.step") or name.startswith("### "):
                                totals[name] = totals.get(name, 0.0) + float(getattr(ev, "device_time_total", 0) or 0)
                        for name, us in sorted(totals.items()):
                            logger.info("marker %s: GPU time %.1f ms per step (steps %d..%d)", name, us / 1000.0 / nsteps, mk_prof_a, mk_prof_b)
                        mk_prof = None
                    should_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    should_log_timer = self.rank_timer_every > 0 and self.global_step % self.rank_timer_every == 0
                    should_gather_metrics = (
                        self.gather_metrics_every_step
                        or should_log
                        or should_log_timer
                    )
                    metrics_start = time.perf_counter()
                    global_loss = None
                    global_loss_metrics = {}
                    global_grad_norm = None
                    if should_gather_metrics:
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        for key, value in loss_dict.items():
                            metric_tensor = torch.as_tensor(
                                value,
                                device=loss.device,
                                dtype=torch.float32,
                            ).detach().reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        grad_norm_tensor = torch.as_tensor(
                            grad_norm,
                            device=loss.device,
                            dtype=torch.float32,
                        ).detach().reshape(1)
                        global_grad_norm = float(
                            self.accelerator.gather(grad_norm_tensor).mean().item()
                        )
                    self._rank_timer_sync(timer_active)
                    step_timings["metrics"] = time.perf_counter() - metrics_start
                    step_timings["total"] = time.perf_counter() - step_start

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    timer_names = None
                    timer_stats = None
                    if should_log_timer:
                        timer_names, timer_stats = self._gather_rank_timer_stats(step_timings, loss.device)

                    if should_log and self.accelerator.is_main_process:
                        if global_loss is None or global_grad_norm is None:
                            raise RuntimeError(
                                "Logging step reached without gathered training metrics."
                            )
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        samples_per_sec = (
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            samples_per_sec,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_sec,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if should_log_timer and self.accelerator.is_main_process:
                        logger.info(
                            "[timer] step=%d rank-wise %s",
                            self.global_step,
                            self._format_rank_timer_stats(timer_names, timer_stats),
                        )

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d samples=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["num_samples"],
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/num_samples": int(metrics["num_samples"]),
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    # Short performance probes include one-time Triton/FLA
                    # compilation in their first steps.  Reset only the speed
                    # clock after an explicit warm-up so variant selection is
                    # based on steady-state work.  This does not touch model,
                    # optimizer, scheduler, sampler, or RNG state.
                    if (
                        speed_warmup_steps > 0
                        and self.global_step == speed_reset_step
                    ):
                        self.run_start_step = self.global_step
                        self.run_start_time = time.perf_counter()

                    if self.global_step >= self.max_steps:
                        self._log_cuda_peak_memory()
                        # When the final step is also a periodic save boundary,
                        # the checkpoint above already contains the exact same
                        # weights/state.  Avoid rewriting the multi-GB files a
                        # second time before returning.
                        if self.save_final_checkpoint and not (
                            self.save_every > 0
                            and self.global_step % self.save_every == 0
                        ):
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            if self.save_final_checkpoint or (
                                self.save_every > 0
                                and self.global_step % self.save_every == 0
                            ):
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                            else:
                                logger.info(
                                    "[done] max_steps reached step=%d; disposable run "
                                    "requested save_final_checkpoint=false",
                                    self.global_step,
                                )
                        return

        self._log_cuda_peak_memory()
        ckpt_info = self.save_checkpoint() if self.save_final_checkpoint else None
        if self.accelerator.is_main_process:
            if ckpt_info is None:
                logger.info(
                    "[done] training finished step=%d; disposable run requested "
                    "save_final_checkpoint=false",
                    self.global_step,
                )
            else:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        
