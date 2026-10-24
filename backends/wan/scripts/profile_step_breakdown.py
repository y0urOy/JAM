"""Per-component wall-time breakdown of one training step (single GPU, no optimizer).

Answers: real time of action expert vs video branch vs VAE, to check the FLOPs
table (video 3.84 vs action 0.40 TFLOPs, 9.6x) against measured time ratios.

Attribution: MoT fuses attention, but the big GEMMs are per-stream and shape-
distinguishable: video qkv/o = [*,3072]x[3072,3072], video ffn = 3072x14336;
action qkv = [*,1024]x[1024,*], action ffn = 1024x4096. Fused attention
(QK^T/AV over mixed tokens) is reported separately. VAE isolated via a
record_function wrapper around build_inputs.

Run (1 GPU): python scripts/profile_step_breakdown.py task=<task> [+prof.cached=true]
"""
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jam_wan.utils.config_resolvers import register_default_resolvers  # noqa: E402

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    ds = instantiate(cfg.data.train)
    # +prof.stride=<n> spaces the 16 profiled samples n indices apart (default 997);
    # use a small stride when the cached run must stay inside the first cache shard.
    stride = int(cfg.get("prof", {}).get("stride", 997)) if cfg.get("prof") is not None else 997
    picks = [ds._get(i) for i in range(0, 16 * stride, stride)]
    batch = torch.utils.data.default_collate(picks)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda")
    # replicate trainer's grad setup
    model.eval(); model.requires_grad_(False)
    model.dit.train(); model.dit.requires_grad_(True)
    for name in ("proprio_encoder", "foresight_projectors"):
        m = getattr(model, name, None)
        if m is not None:
            m.train(); m.requires_grad_(True)

    if getattr(model, "foresight_use_target_cache", False):
        # attach cached targets like ForesightCacheDataset would
        from jam_wan.datasets.foresight_cache import ForesightCacheReader
        roots = model_cfg["foresight"]["target_cache_roots"]
        rd = ForesightCacheReader(roots)
        rows = [rd.layers.index(int(L)) for L in model.foresight_encoder_layers]
        cache = torch.stack([rd.get(i)[rows] for i in range(0, 16 * stride, stride)])
        batch["foresight_enc_cache"] = cache

    orig_build = model.build_inputs
    def build_wrapped(*a, **k):
        with record_function("### VAE+build_inputs"):
            return orig_build(*a, **k)
    model.build_inputs = build_wrapped

    def one_step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = model.training_loss(batch)
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(3):
        one_step()
    torch.cuda.synchronize()

    # wall-clock per step (fwd+bwd, no optimizer) without profiler overhead
    import time
    t0 = time.perf_counter()
    for _ in range(6):
        one_step()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / 6 * 1000

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        for _ in range(6):
            one_step()
        torch.cuda.synchronize()

    # Attribute via ATEN leaf ops (they carry input_shapes AND aggregate their
    # kernels' device time). Leaf set avoids parent/child double counting.
    ka = prof.key_averages(group_by_input_shape=True)
    LEAF = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
            "aten::convolution", "aten::_convolution", "aten::cudnn_convolution",
            "aten::scaled_dot_product_attention")
    buckets = {"video_gemm": 0.0, "action_gemm": 0.0, "fused_attn": 0.0,
               "conv_vae": 0.0, "other_leaf": 0.0}
    fbuckets = {k: 0.0 for k in buckets}
    VIDEO_DIMS = {3072, 14336}
    ACTION_DIMS = {1024, 4096}
    rows = []
    total_cuda = 0.0
    kernel_cuda = 0.0
    from torch.autograd import DeviceType
    for e in ka:
        total_cuda += (getattr(e, "self_device_time_total", 0) or 0)
        if getattr(e, "device_type", None) == DeviceType.CUDA:
            kernel_cuda += (getattr(e, "self_device_time_total", 0) or 0)
    for e in ka:
        key = e.key or ""
        if not key.startswith(LEAF):
            continue
        dev_us = getattr(e, "device_time_total", 0) or 0
        if dev_us <= 0:
            continue
        dims = set()
        for shp in (e.input_shapes or []):
            for d in (shp or []):
                if isinstance(d, int):
                    dims.add(d)
        fl = float(getattr(e, "flops", 0) or 0)
        if "scaled_dot_product" in key:
            b = "fused_attn"; tag = "attn"
        elif "convolution" in key.lower():
            b = "conv_vae"; tag = "vae/conv"
        elif dims & VIDEO_DIMS:
            b = "video_gemm"; tag = "video"
        elif dims & ACTION_DIMS:
            b = "action_gemm"; tag = "action"
        else:
            b = "other_leaf"; tag = "other"
        buckets[b] += dev_us
        fbuckets[b] += fl
        rows.append((dev_us, tag, key[:60], str(e.input_shapes)[:110]))

    n = 6.0
    print("\n========== per-step CUDA time breakdown (ms, avg of 6 steps) ==========")
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:12s} {v/1000/n:9.1f} ms")
    tot = sum(buckets.values())
    print(f"  {'LEAF SUM':12s} {tot/1000/n:9.1f} ms")
    print(f"  {'ALL-CUDA':12s} {total_cuda/1000/n:9.1f} ms   (device total incl. non-leaf)")
    print(f"  {'KERNEL SUM':12s} {kernel_cuda/1000/n:9.1f} ms   (CUDA kernel events only, no double counting)")
    print(f"  {'WALL-CLOCK':12s} {wall_ms:9.1f} ms   (fwd+bwd per step, synchronized, no profiler)")

    # Pure-CUDA-activity anchor: with only device events recorded, the device total is the GPU
    # kernel time per step without the repeated counting of CPU+CUDA mode (~3.1-3.3x there).
    with profile(activities=[ProfilerActivity.CUDA]) as prof2:
        for _ in range(6):
            one_step()
        torch.cuda.synchronize()
    gpu_ms = sum((getattr(e, "self_device_time_total", 0) or 0) for e in prof2.key_averages()) / 1000 / n
    print(f"  {'GPU TIME':12s} {gpu_ms:9.1f} ms   (pure-CUDA-activity profile, per step)")
    # inclusive device time of the VAE+build_inputs wrapper (one parent event, no nesting inflation)
    for e in ka:
        if (e.key or "").startswith("### VAE+build_inputs"):
            print(f"  {'VAE+build':12s} {(getattr(e, 'device_time_total', 0) or 0)/1000/n:9.1f} ms   (inclusive GPU time of VAE encoding and input build, CPU+CUDA mode)")
    print("\n========== measured FLOPs per step (TFLOPs, fwd+bwd, avg of 6) ==========")
    for k, v in sorted(fbuckets.items(), key=lambda x: -x[1]):
        eff = (v / n / 1e12) / (buckets[k] / 1e6 / n) if buckets[k] > 0 else 0
        print(f"  {k:12s} {v/n/1e12:9.3f} TFLOP   ({eff:6.1f} TFLOPS effective)")
    print(f"  {'SUM':12s} {sum(fbuckets.values())/n/1e12:9.3f} TFLOP")
    print("\n========== top-25 kernels ==========")
    for cuda_us, tag, key, shp in sorted(rows, reverse=True)[:25]:
        print(f"  {cuda_us/1000/n:8.1f} ms [{tag:6s}] {key}  {shp}")


if __name__ == "__main__":
    main()
