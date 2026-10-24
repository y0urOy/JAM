"""CPU unit tests for the foresight target cache (no GPU, no real data).

Run: python scripts/test_foresight_cache.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jam_wan.datasets.foresight_cache import (  # noqa: E402
    ForesightCacheDataset, ForesightCacheReader, ShardWriter, shard_name, write_meta,
)

L_LAYERS, SEQ, DIM = [14, 16], 12, 64
N, SHARD = 10, 4  # 3 shards: [0-3], [4-7], [8-9]


class FakeBase(torch.utils.data.Dataset):
    """Mimics RobotVideoDataset: has _get(idx); idx 7 always fails (tests fallback)."""

    def __len__(self):
        return N

    def _get(self, idx):
        if idx == 7:
            raise RuntimeError("simulated decode failure")
        return {"video": torch.full((2, 2), float(idx)), "idx_check": idx}

    def __getitem__(self, idx):
        return self._get(idx)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fs_cache_test_"))
    root_a, root_b = tmp / "a", tmp / "b"
    root_a.mkdir(); root_b.mkdir()

    torch.manual_seed(0)
    records = {i: torch.randn(len(L_LAYERS), SEQ, DIM).to(torch.bfloat16) for i in range(N)}

    # Write shards striped across two roots (0->a, 1->b, 2->a).
    for k, start in enumerate(range(0, N, SHARD)):
        size = min(SHARD, N - start)
        root = root_a if k % 2 == 0 else root_b
        w = ShardWriter(root, start, size)
        for i in range(start, start + size):
            w.add(i, records[i])
        w.finalize()
    for r in (root_a, root_b):
        write_meta(r, layers=L_LAYERS, seq_len=SEQ, dim=DIM, num_samples=N, shard_size=SHARD)

    # 1) bit-exact round trip across roots
    reader = ForesightCacheReader([root_a, root_b])
    assert reader.missing_shards() == [], reader.missing_shards()
    for i in range(N):
        got = reader.get(i)
        assert got.dtype == torch.bfloat16
        assert torch.equal(got.view(torch.uint16), records[i].view(torch.uint16)), f"bit mismatch @ {i}"
    print("PASS 1: bf16+zstd round-trip is bit-exact across 2 roots")

    # 2) layer subset + ordering ([16] must select row 1)
    ds = ForesightCacheDataset(FakeBase(), [root_a, root_b], layers=[16])
    item = ds[3]
    assert torch.equal(item["foresight_enc_cache"].view(torch.uint16),
                       records[3][1:2].view(torch.uint16))
    assert item["idx_check"] == 3
    print("PASS 2: layer-subset selection ([16] -> row 1) is bit-exact")

    # 3) fallback consistency: idx 7 fails -> substituted, but data and cache
    #    must come from the SAME substituted index.
    item = ds[7]
    sub = item["idx_check"]
    assert sub != 7
    assert torch.equal(item["foresight_enc_cache"].view(torch.uint16),
                       records[sub][1:2].view(torch.uint16)), "fallback data/cache index mismatch!"
    print(f"PASS 3: decode-failure fallback keeps data/cache consistent (7 -> {sub})")

    # 4) default_collate stacks the extra key (bf16 batch)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    assert batch["foresight_enc_cache"].shape == (2, 1, SEQ, DIM)
    assert batch["foresight_enc_cache"].dtype == torch.bfloat16
    print("PASS 4: default_collate produces [B, L, seq, D] bf16")

    # 5) guards: wrong layer, wrong dataset length, incomplete cache
    try:
        ForesightCacheDataset(FakeBase(), [root_a, root_b], layers=[12])
        raise AssertionError("expected layer-mismatch error")
    except RuntimeError as e:
        assert "cache has layers" in str(e)
    (root_a / (shard_name(0) + ".bin")).unlink()
    try:
        ForesightCacheDataset(FakeBase(), [root_a, root_b], layers=[16])
        raise AssertionError("expected incomplete-cache error")
    except RuntimeError as e:
        assert "incomplete" in str(e)
    print("PASS 5: guards (missing layer / missing shard) fire correctly")

    # 6) compression sanity on realistic-ish data
    import os
    total_raw = sum(r.numel() * 2 for r in records.values())
    total_disk = sum(p.stat().st_size for r in (root_a, root_b) for p in r.glob("shard_*.bin"))
    print(f"PASS 6: zstd ratio on random bf16 = {total_raw / max(total_disk, 1):.2f}x "
          f"(random data is worst-case; real features compress better)")

    # 7) Compact VAE-direct cache stores only future offsets [1,2] at raw
    #    192-D, then restores the full 3-frame temporal grid on read.
    compact_root = tmp / "compact_vae"
    compact_root.mkdir()
    compact_tpf, compact_dim = 3, 192
    compact_records = {
        i: torch.randn(1, 2 * compact_tpf, compact_dim).to(torch.bfloat16)
        for i in range(N)
    }
    for start in range(0, N, SHARD):
        size = min(SHARD, N - start)
        writer = ShardWriter(compact_root, start, size)
        for i in range(start, start + size):
            writer.add(i, compact_records[i])
        writer.finalize()
    write_meta(
        compact_root,
        layers=[0],
        seq_len=2 * compact_tpf,
        dim=compact_dim,
        num_samples=N,
        shard_size=SHARD,
        extra={
            "tokens_per_frame": compact_tpf,
            "target_type": "vae_direct",
            "stored_offsets": [1, 2],
            "num_target_frames": 3,
            "representation": "raw_vae_patch_before_isometric_lift",
        },
    )
    compact_reader = ForesightCacheReader([compact_root])
    compact_got = compact_reader.get(2).view(1, 3, compact_tpf, compact_dim)
    assert torch.count_nonzero(compact_got[:, 0]) == 0
    assert torch.equal(
        compact_got[:, 1:].reshape_as(compact_records[2]).view(torch.uint16),
        compact_records[2].view(torch.uint16),
    )
    compact_ds = ForesightCacheDataset(
        FakeBase(),
        [compact_root],
        layers=[0],
        target_type="vae_direct",
        offsets=[1, 2],
    )
    assert compact_ds[2]["foresight_enc_cache"].shape == (
        1,
        3 * compact_tpf,
        compact_dim,
    )
    try:
        ForesightCacheDataset(
            FakeBase(),
            [compact_root],
            layers=[0],
            target_type="vae_direct",
            offsets=[0, 1, 2],
        )
        raise AssertionError("expected compact-offset mismatch")
    except RuntimeError as error:
        assert "additionally wants" in str(error)
    print("PASS 7: compact VAE-direct [1,2] cache expands offsets exactly and guards mismatch")

    # 8) A separate current-frame VAE-latent cache can be attached to the same
    #    exact sample index.  It is reshaped back to [C,T,H,W] bit-for-bit and
    #    default_collate adds only the batch dimension.
    input_root = tmp / "input_vae_latent"
    input_root.mkdir()
    latent_shape = (2, 1, 2, 2)
    input_records = {
        i: torch.randn(1, 4, 2).to(torch.bfloat16)
        for i in range(N)
    }
    for start in range(0, N, SHARD):
        size = min(SHARD, N - start)
        writer = ShardWriter(input_root, start, size)
        for i in range(start, start + size):
            writer.add(i, input_records[i])
        writer.finalize()
    write_meta(
        input_root,
        layers=[0],
        seq_len=4,
        dim=2,
        num_samples=N,
        shard_size=SHARD,
        extra={
            "target_type": "input_vae_latent",
            "latent_shape": list(latent_shape),
            "representation": "frozen_policy_vae_current_latent",
            "source_split": "train",
        },
    )
    dual_ds = ForesightCacheDataset(
        FakeBase(),
        [compact_root],
        layers=[0],
        target_type="vae_direct",
        offsets=[1, 2],
        input_latent_roots=[input_root],
    )
    dual_item = dual_ds[2]
    assert dual_item["input_latents_cache"].shape == latent_shape
    assert torch.equal(
        dual_item["input_latents_cache"].reshape_as(input_records[2]).view(torch.uint16),
        input_records[2].view(torch.uint16),
    )
    dual_batch = torch.utils.data.default_collate([dual_ds[0], dual_ds[1]])
    assert dual_batch["input_latents_cache"].shape == (2,) + latent_shape
    print("PASS 8: target + current-input caches remain index-aligned and bit-exact")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
