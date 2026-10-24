#!/usr/bin/env python3
"""CPU regressions for DINOv2-B + cached Qwen3 + ActionDiTDino12."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import torch

from jam_multimodal.models.backbones.jam import JAMPolicy


def _bare_qwen_cache_model(cache_dir: Path) -> JAMPolicy:
    model = JAMPolicy.__new__(JAMPolicy)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.external_ssl_text_condition_source = "qwen3_4b_flux2_cached"
    model.external_ssl_qwen_text_cache_dir = str(cache_dir)
    model.external_ssl_qwen_context_len = 8
    model.external_ssl_qwen_hidden_dim = 16
    object.__setattr__(model, "_external_ssl_text_cache", {})
    return model


def check_prompt_cache_and_padding_trim() -> None:
    prompt = "A video recorded from a robot's point of view executing the following instruction: test"
    with tempfile.TemporaryDirectory() as temporary:
        cache_dir = Path(temporary)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        context = torch.arange(8 * 16, dtype=torch.bfloat16).reshape(8, 16)
        mask = torch.tensor(
            [True, True, True, False, False, False, False, False]
        )
        torch.save(
            {
                "text_hidden_states": context,
                "text_attention_mask": mask,
            },
            cache_dir / f"{digest}.qwen3_flux2_len8.pt",
        )
        model = _bare_qwen_cache_model(cache_dir)
        loaded, loaded_mask = model._read_external_ssl_qwen_text_cache(prompt)
        assert tuple(loaded.shape) == (1, 3, 16)
        assert tuple(loaded_mask.shape) == (1, 3)
        assert bool(loaded_mask.all())
        assert torch.equal(loaded[0], context[:3].float())

        # Training receives fixed-width batched cache tensors from the dataset;
        # columns masked for every sample must be removed without changing any
        # valid token or its position.
        batched = torch.stack((context, context + 1), dim=0)
        batched_mask = torch.tensor(
            [
                [True, True, True, False, False, False, False, False],
                [True, True, True, True, True, False, False, False],
            ]
        )
        encoded, encoded_mask = model._encode_ovis_u1_text(
            {
                "text_hidden_states": batched,
                "text_attention_mask": batched_mask,
            }
        )
        assert tuple(encoded.shape) == (2, 5, 16)
        assert tuple(encoded_mask.shape) == (2, 5)
        assert torch.equal(encoded, batched[:, :5].float())
        assert torch.equal(encoded_mask, batched_mask[:, :5])


def main() -> None:
    check_prompt_cache_and_padding_trim()
    print("DINO QWEN3 NATIVE ACTION TEXT CHECKS PASSED")


if __name__ == "__main__":
    main()
