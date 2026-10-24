#!/usr/bin/env python3
"""CPU regression checks for the FLUX.2 foresight plumbing."""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The toy blocks only need the public apply_rope symbol used by MoT.
flux2_pkg = types.ModuleType("flux2")
flux2_model = types.ModuleType("flux2.model")
flux2_model.apply_rope = lambda q, k, pe: (q, k)
flux2_pkg.model = flux2_model
sys.modules["flux2"] = flux2_pkg
sys.modules["flux2.model"] = flux2_model

from jam_multimodal.models.backbones.foresight_flux2 import attach_flux2_foresight
from jam_multimodal.models.backbones.jam import JAMPolicy
from jam_multimodal.models.backbones.mot import MoT


def _heads(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


class ToyDoubleBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.bias = nn.Parameter(torch.full((4,), scale))

    def _prepare_qkv(self, img, txt, img_pe, txt_pe, img_mod, txt_mod):
        del img_pe, txt_pe, img_mod, txt_mod
        joint = torch.cat([txt, img], dim=1)
        qkv = _heads(joint)
        return qkv, qkv, qkv, None, txt.shape[1], None

    def _apply_residuals(self, img, txt, img_attn, txt_attn, mods):
        del mods
        return img + 0.1 * img_attn + self.bias, txt + 0.1 * txt_attn + self.bias


class ToySingleBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.bias = nn.Parameter(torch.full((4,), scale))

    def _qkv(self, x, mod):
        del mod
        qkv = _heads(x)
        return qkv, qkv, qkv, x, torch.ones_like(x)

    def _out(self, residual, mixed, mlp, gate):
        del mlp, gate
        return residual + 0.1 * mixed + self.bias


class ToyActionDouble(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.bias = nn.Parameter(torch.full((4,), scale))

    def prepare_qkv(self, action, action_pe, mod):
        del action_pe, mod
        return {"q": action, "k": action, "v": action, "residual": action}

    def apply_post(self, mixed, state):
        return state["residual"] + 0.1 * mixed + self.bias


class ToyActionSingle(ToyActionDouble):
    pass


class ToyTransformer(nn.Module):
    @staticmethod
    def pe_embedder(ids):
        return ids[:, None, :, :1]


class ToyVideoExpert(nn.Module):
    block_protocol = "flux2"

    def __init__(self):
        super().__init__()
        self.transformer = ToyTransformer()
        self.double_blocks = nn.ModuleList([ToyDoubleBlock(0.01), ToyDoubleBlock(0.02)])
        self.single_blocks = nn.ModuleList([ToySingleBlock(0.03), ToySingleBlock(0.04)])
        self.double_layers = len(self.double_blocks)
        self.single_layers = len(self.single_blocks)
        self.hidden_dim = 4
        self.num_heads = 1
        self.num_kv_heads = 1
        self.attn_head_dim = 4

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)


class ToyActionExpert(nn.Module):
    block_protocol = "flux2"

    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([ToyActionDouble(0.01), ToyActionDouble(0.02)])
        self.single_blocks = nn.ModuleList([ToyActionSingle(0.03), ToyActionSingle(0.04)])
        self.num_heads = 1
        self.num_kv_heads = 1
        self.attn_head_dim = 4

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)


class ToyPolicy(nn.Module):
    def __init__(self, video: ToyVideoExpert, mot: MoT):
        super().__init__()
        self.video_expert = video
        self.mot = mot
        self.stack = "flux2"


class IdentityProjector(nn.Module):
    @staticmethod
    def trunk(x):
        return x

    @staticmethod
    def head(x, idx):
        del idx
        return x


def _inputs():
    torch.manual_seed(7)
    txt = torch.randn(2, 2, 4)
    img = torch.randn(2, 4, 4)
    action = torch.randn(2, 3, 4)
    txt_pe = torch.zeros(2, 1, txt.shape[1], 1)
    img_pe = torch.zeros(2, 1, img.shape[1], 1)
    action_ids = torch.zeros(2, action.shape[1], 4)
    video_len = txt.shape[1] + img.shape[1]
    full_len = video_len + action.shape[1]
    return txt, img, action, txt_pe, img_pe, action_ids, video_len, full_len


def main() -> None:
    video = ToyVideoExpert()
    action_expert = ToyActionExpert()
    mot = MoT(
        mixtures={"video": video, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    )
    txt, img, action, txt_pe, img_pe, action_ids, video_len, full_len = _inputs()

    # Student capture after global layer 3 (first single-stream block).
    mot._foresight_capture_layers = {3}
    out = mot(
        embeds_all={"video": {"txt": txt, "img": img}, "action": action},
        attention_mask={
            "double_joint": torch.ones(2, full_len, full_len, dtype=torch.bool),
            "single": torch.ones(2, full_len, full_len, dtype=torch.bool),
        },
        freqs_all={"video": {"txt": txt_pe, "img": img_pe}},
        context_all={"video": None, "action": {"ids": action_ids}},
        t_mod_all={
            "video": {"double_img": None, "double_txt": None, "single": None},
            "action": {"double_img": None, "single": None},
        },
    )
    captured_student = mot._foresight_video_hiddens[3]
    assert captured_student.shape == img.shape
    captured_student.sum().backward()
    assert video.double_blocks[0].bias.grad is not None
    assert out["video"]["img"].shape == img.shape

    # Frozen teacher early-exit must preserve the exact requested feature.
    teacher = copy.deepcopy(video)
    video_mask = {
        "double_joint": torch.ones(2, video_len, video_len, dtype=torch.bool),
        "single": torch.ones(2, video_len, video_len, dtype=torch.bool),
    }
    kwargs = dict(
        video_tokens={"txt": txt.detach(), "img": img.detach()},
        video_freqs={"txt": txt_pe, "img": img_pe},
        video_t_mod={"double_img": None, "double_txt": None, "single": None},
        attention_mask=video_mask,
        capture_layers={3},
        video_expert=teacher,
    )
    full = mot.forward_flux2_video_only(**kwargs, early_exit_after_last_capture=False)
    early = mot.forward_flux2_video_only(**kwargs, early_exit_after_last_capture=True)
    assert full["executed_layers"] == 4
    assert early["executed_layers"] == 3
    assert torch.equal(full["captured"][3], early["captured"][3])

    # Attachment keeps the teacher out of state_dict and registers projector.
    policy = ToyPolicy(video, mot)
    attach_flux2_foresight(
        policy,
        {
            "enabled": True,
            "teacher_mode": "frozen_pretrained",
            "layer": 3,
            "encoder_layer": 3,
            "offsets": [1],
            "lambda_foresight": 0.3,
            "projector_hidden_dim": 4,
            "projector_ffn_dim": 8,
            "projector_num_layers": 1,
            "projector_num_heads": 1,
        },
    )
    assert policy.foresight_layer == 3
    assert policy.foresight_offsets == [1]
    assert hasattr(policy.mot, "foresight_projector")
    assert all(not parameter.requires_grad for parameter in policy.foresight_teacher.parameters())
    assert not any("foresight_teacher" in key for key in policy.state_dict())

    # FLUX.2 packs image tokens [current/ref, future/target]. Verify offset=1
    # selects the latter, rather than carrying over Ovis's opposite ordering.
    future = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
    )
    current = torch.tensor(
        [[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]],
    )
    teacher_img = torch.cat([current, future], dim=1)

    class ControlledTeacherExpert:
        @staticmethod
        def pre_dit(**kwargs):
            del kwargs
            return {
                "tokens": {"txt": torch.zeros(1, 1, 4), "img": teacher_img},
                "freqs": {"txt": torch.zeros(1, 1, 1, 1), "img": torch.zeros(1, 1, 4, 1)},
                "t_mod": {"double_img": None, "double_txt": None, "single": None},
                "txt_len": 1,
                "cond_len": 2,
                "target_len": 2,
                "text_mask": torch.ones(1, 1, dtype=torch.bool),
            }

    controlled_teacher = types.SimpleNamespace(expert=ControlledTeacherExpert())
    object.__setattr__(policy, "foresight_teacher", controlled_teacher)
    policy.device = torch.device("cpu")
    policy.foresight_layer = 3
    policy.foresight_encoder_layer = 3
    policy.foresight_teacher_early_exit = True
    policy.foresight_offsets = [1]
    policy.mot.foresight_projector = IdentityProjector()

    original_video_only = policy.mot.forward_flux2_video_only

    def controlled_video_only(self, **kwargs):
        assert all(mask.is_contiguous() for mask in kwargs["attention_mask"].values())
        return {"captured": {3: teacher_img}}

    policy.mot.forward_flux2_video_only = types.MethodType(controlled_video_only, policy.mot)
    policy._build_mot_attention_mask_flux2 = types.MethodType(
        JAMPolicy._build_mot_attention_mask_flux2.__wrapped__, policy
    )
    loss = JAMPolicy._compute_flux2_foresight_loss(
        policy,
        inputs={
            "target_latent": torch.zeros(1, 2, 4),
            "text_hidden_states": torch.zeros(1, 1, 4),
            "text_attention_mask": torch.ones(1, 1, dtype=torch.bool),
            "ref_image_latents": torch.zeros(1, 2, 4),
            "target_img_ids": torch.zeros(1, 2, 4),
            "ref_img_ids": torch.zeros(1, 2, 4),
        },
        video_pre={"cond_len": 2},
        captured={3: future.clone().requires_grad_(True)},
        batch_size=1,
    )
    policy.mot.forward_flux2_video_only = original_video_only
    assert float(loss) < 1e-6, loss
    print("[ok] FLUX.2 foresight capture, early-exit, and attachment checks passed")


if __name__ == "__main__":
    # Avoid inheriting an eval-only shell flag into this training regression.
    os.environ.pop("JAM_FORESIGHT_NO_TEACHER", None)
    main()
