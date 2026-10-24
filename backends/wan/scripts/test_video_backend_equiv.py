"""Bit-exact check: JAM_VIDEO_BACKEND=pyav vs torchcodec (LIBERO AV1 mp4s).

Same dataset config, same sample indices; the ONLY difference is the video
backend env var (consumed once at LeRobotDataset construction). If pixels are
bitwise identical, pyav is a drop-in escape hatch from the dav1d-1.2.1 stack.
AV1 decode is spec-deterministic; the real risk is yuv420p->rgb conversion
(FFmpeg 7 swscale vs PyAV/FFmpeg 8 path), hence this test.

Run on CPU: python scripts/test_video_backend_equiv.py
"""
import os
import random
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jam_wan.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()

TASK = "libero_bplus_na_downpred118_lam06_single18_2cam224_1e-4"


def build(backend):
    os.environ["JAM_VIDEO_BACKEND"] = backend
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    ds = instantiate(cfg.data.train)
    # verify the backend actually took effect
    inner = ds.lerobot_dataset.multi_dataset._datasets[0]
    actual = getattr(inner, "video_backend", "?")
    print(f"built dataset with JAM_VIDEO_BACKEND={backend} -> video_backend={actual}", flush=True)
    assert str(actual) == backend, f"backend override did not take: wanted {backend}, got {actual}"
    return ds


def main():
    ds_tc = build("torchcodec")
    ds_av = build("pyav")
    n = len(ds_tc)
    assert n == len(ds_av)

    random.seed(0)
    picks = [0, n // 4, n // 2, 3 * n // 4, n - 1] + [random.randrange(n) for _ in range(7)]

    worst = 0.0
    n_bit = 0
    for idx in picks:
        va = ds_tc._get(idx)["video"].float()
        vb = ds_av._get(idx)["video"].float()
        assert va.shape == vb.shape, f"shape mismatch @ {idx}: {va.shape} vs {vb.shape}"
        if torch.equal(va, vb):
            n_bit += 1
            print(f"idx {idx}: BITWISE identical  shape={tuple(va.shape)}", flush=True)
        else:
            d = (va - vb).abs()
            mx = d.max().item()
            worst = max(worst, mx)
            frac = (d > 0).float().mean().item()
            # video is normalized to [-1,1]; 1 uint8 LSB = 2/255 ~ 7.84e-3
            print(f"idx {idx}: DIFF max={mx:.3e} ({mx*255/2:.2f} uint8-LSB) "
                  f"nonzero_frac={frac:.4f}", flush=True)

    print(f"\n{n_bit}/{len(picks)} samples bitwise identical; worst diff = "
          f"{worst:.3e} ({worst*255/2:.2f} uint8-LSB)")
    if n_bit == len(picks):
        print("RESULT: BIT-EXACT — pyav is a drop-in replacement")
    elif worst <= 2 * 2 / 255 + 1e-9:
        print("RESULT: NOT bit-exact but within ±2 uint8 LSB (colorspace rounding) — "
              "safe for training, NOT run-comparable bit-for-bit")
    else:
        print("RESULT: SUBSTANTIAL pixel differences — investigate before switching")


if __name__ == "__main__":
    main()
