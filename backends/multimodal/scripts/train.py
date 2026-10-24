import os

import hydra
import torch
from omegaconf import DictConfig

from jam_multimodal.runtime import run_training
from jam_multimodal.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Debug-only: env-gated NaN-gradient localizer. No-op unless explicitly set.
    if os.environ.get("JAM_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    # Debug-only: force a specific SDPA backend (e.g. "math") to bisect whether the
    # flash / mem-efficient SDPA backward on a given GPU software stack emits NaN grads.
    _sdpa = os.environ.get("JAM_SDPA_BACKEND", "").strip().lower()
    if _sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp("flash" in _sdpa)
        torch.backends.cuda.enable_mem_efficient_sdp("mem" in _sdpa)
        torch.backends.cuda.enable_math_sdp("math" in _sdpa)
    run_training(cfg)


if __name__ == "__main__":
    main()
