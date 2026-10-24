"""video_future_frames dataset verification.

Dataset-level only (no GPU needed). Run from repo root:
  python scripts/test_video_future_frames.py

  A: vff=0 regression — shapes identical to the historical pipeline
     (video [C,9,H,W], action 32, image_is_pad 9, proprio 33-grid, obs window 33)
  B: vff=4 decoupling — video [C,13,H,W] / image_is_pad 13, action STILL 32
  C: BITWISE — vff=4's first 9 frames == vff=0's video; is_pad[:9] equal;
     action/proprio/context bitwise equal (extension only APPENDS)
  D: padded-tail sample — appended future frames carry pad=True correctly
(D: instantiate goes through hydra task compose WITH the real processor + text
 cache, so the processor.image_obs_steps pitfall is exercised.)
"""
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jam_wan.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()

TASK = "libero_bplus_na_downpred118_lam06_single18_2cam224_1e-4"


def main():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    ds0 = instantiate(cfg.data.train)                          # vff=0 (config default)
    ds4 = instantiate(cfg.data.train, video_future_frames=4)   # vff=4

    # grid-level invariants
    assert len(ds0.video_sample_indices) == 9 and ds0.num_action_video_frames == 9
    assert len(ds4.video_sample_indices) == 13 and ds4.num_action_video_frames == 9
    assert ds4.video_sample_indices[-1] == 48  # 33 + 4*4 - 1
    assert len(ds0) == len(ds4)
    print(f"grids OK: vff0={len(ds0.video_sample_indices)}f vff4={len(ds4.video_sample_indices)}f "
          f"(action-window frames both {ds0.num_action_video_frames}); len={len(ds0)}", flush=True)

    n = len(ds0)
    normal_idx, pad_idx, i = [], [], n // 3
    while (len(pad_idx) < 2 or len(normal_idx) < 2) and i < n:
        s = ds4._get(i)
        if bool(s["image_is_pad"].any()):
            if len(pad_idx) < 2:
                pad_idx.append(i)
        elif len(normal_idx) < 2:
            normal_idx.append(i)
        i += 997
    picks = normal_idx[:2] + pad_idx[:2]
    print(f"test idx: normal={normal_idx[:2]} padded={pad_idx[:2]}", flush=True)

    ok = True
    for idx in picks:
        a, b = ds0._get(idx), ds4._get(idx)

        # A: vff=0 shapes = historical
        assert a["video"].ndim == 4 and a["video"].shape[1] == 9, a["video"].shape
        assert a["action"].shape[0] == 32 and a["image_is_pad"].shape[0] == 9
        assert a["proprio_is_pad"].shape[0] == 33

        # B: vff=4 video/pad extended, action/proprio NOT
        assert b["video"].shape[1] == 13, b["video"].shape
        assert b["image_is_pad"].shape[0] == 13
        assert b["action"].shape[0] == 32 and b["action_is_pad"].shape[0] == 32
        assert b["proprio"].shape[0] == 32 and b["proprio_is_pad"].shape[0] == 33

        # C: bitwise — extension only APPENDS
        c_video = torch.equal(b["video"][:, :9].contiguous(), a["video"].contiguous())
        c_pad = torch.equal(b["image_is_pad"][:9], a["image_is_pad"])
        c_rest = (torch.equal(a["action"], b["action"]) and torch.equal(a["proprio"], b["proprio"])
                  and torch.equal(a["context"], b["context"])
                  and torch.equal(a["action_is_pad"], b["action_is_pad"]))
        print(f"idx {idx}: video[:9] bitwise={c_video} pad[:9]={c_pad} action/proprio/context={c_rest} "
              f"future_pad={b['image_is_pad'][9:].tolist()}", flush=True)
        ok &= c_video and c_pad and c_rest

    # D: at least one padded pick must show pad=True in the appended tail
    tail_pads = [bool(ds4._get(idx)["image_is_pad"][9:].any()) for idx in pad_idx[:2]]
    print(f"padded-tail future-frame pads: {tail_pads}", flush=True)
    ok &= any(tail_pads) if pad_idx else True

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
