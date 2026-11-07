"""PharmacophoreEncoder: distance-aware message passing over feature points."""

from __future__ import annotations

import math

import torch

from src.model.architectures.pharmacophore import PharmacophoreEncoder

NODE, EDGE, GLOBAL, OUT = 11, 13, 258, 16


def _batch(counts: list[int], *, has_3d: list[float], seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    k = sum(counts)
    batch = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    # 2D edges: consecutive points of each molecule
    src, dst, offset = [], [], 0
    for c in counts:
        for i in range(c - 1):
            src += [offset + i, offset + i + 1]
            dst += [offset + i + 1, offset + i]
        offset += c
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    return {
        "pharm_x": torch.randn(k, NODE, generator=g),
        "pharm_edge_index": edge_index,
        "pharm_edge_attr": torch.randn(edge_index.size(1), EDGE, generator=g),
        "pharm_pos": torch.randn(k, 3, generator=g) * 3,
        "pharm_global": torch.randn(len(counts), GLOBAL, generator=g),
        "has_3d": torch.tensor(has_3d),
        "pharm_batch": batch,
        "num_molecules": len(counts),
    }


def _encoder() -> PharmacophoreEncoder:
    torch.manual_seed(0)
    return PharmacophoreEncoder(NODE, EDGE, GLOBAL, OUT).eval()


def test_encoder_shapes():
    out = _encoder()(**_batch([4, 0, 6], has_3d=[1.0, 0.0, 1.0]))
    assert out.shape == (3, OUT)
    assert torch.isfinite(out).all()


def test_k0_molecules_get_the_empty_vector():
    enc = _encoder()
    out = enc(**_batch([4, 0, 0], has_3d=[1.0, 0.0, 1.0]))
    assert torch.equal(out[1], out[2])
    assert torch.equal(out[1], enc.empty)


def test_encoder_ignores_positions_without_3d():
    enc = _encoder()
    a = _batch([5, 3], has_3d=[0.0, 0.0])
    b = dict(a, pharm_pos=a["pharm_pos"] + torch.randn_like(a["pharm_pos"]))
    assert torch.allclose(enc(**a), enc(**b))


def test_encoder_uses_positions_with_3d():
    enc = _encoder()
    a = _batch([5], has_3d=[1.0])
    b = dict(a, pharm_pos=a["pharm_pos"] * 1.5)
    assert not torch.allclose(enc(**a), enc(**b))


def test_encoder_is_rotation_invariant():
    enc = _encoder()
    a = _batch([5, 7], has_3d=[1.0, 1.0])
    t = math.radians(37)
    rot = torch.tensor(
        [[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0], [0, 0, 1]]
    )
    b = dict(a, pharm_pos=a["pharm_pos"] @ rot.T + torch.tensor([1.0, -2.0, 0.5]))
    assert torch.allclose(enc(**a), enc(**b), atol=1e-5)


def test_molecules_do_not_see_each_other():
    enc = _encoder()
    a = _batch([4, 6], has_3d=[1.0, 1.0])
    b = dict(a, pharm_x=a["pharm_x"].clone())
    b["pharm_x"][4:] += 1.0  # change molecule 1 only
    out_a, out_b = enc(**a), enc(**b)
    assert torch.allclose(out_a[0], out_b[0])
    assert not torch.allclose(out_a[1], out_b[1])
