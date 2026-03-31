"""
Unit tests for MetaEncoder, RoPE, and ProfileRegressor.
Run with: pytest tests/
"""

import torch
import pytest

from models.meta_encoder import (
    MetaEncoder,
    ProfileRegressor,
    RotaryPositionEmbedding,
    RoPEMultiHeadAttention,
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

LAYER_DIMS = [256] * 8   # D_flow = 256 for all layers (flow-compression backbone)


def make_traj(B=4, layer_dims=LAYER_DIMS):
    return [torch.randn(B, d) for d in layer_dims]


# --------------------------------------------------------------------------- #
# RotaryPositionEmbedding
# --------------------------------------------------------------------------- #

class TestRoPE:
    def test_output_shape_preserved(self):
        rope = RotaryPositionEmbedding(dim=32, max_positions=16)
        q = torch.randn(2, 4, 8, 32)  # [B, heads, seq, dim]
        k = torch.randn(2, 4, 8, 32)
        q_rot, k_rot = rope(q, k, seq_len=8)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions_produce_different_rotations(self):
        rope = RotaryPositionEmbedding(dim=32)
        q = torch.ones(1, 1, 4, 32)
        k = torch.ones(1, 1, 4, 32)
        q_rot, _ = rope(q, k, seq_len=4)
        # Different positions should produce different rotated vectors
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1], atol=1e-5)
        assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 3], atol=1e-5)

    def test_relative_distance_decay(self):
        """Dot product between q and k should decay with position distance."""
        rope = RotaryPositionEmbedding(dim=64)
        # Same vector at all positions
        v = torch.randn(1, 1, 1, 64).expand(1, 1, 8, 64).clone()
        q_rot, k_rot = rope(v, v, seq_len=8)

        # Dot product of position 0 with positions 1, 2, 4, 7
        q0 = q_rot[0, 0, 0]
        dots = []
        for pos in [1, 2, 4, 7]:
            dots.append((q0 * k_rot[0, 0, pos]).sum().item())

        # Should generally decay with distance (not strictly monotonic but
        # the nearest should be highest)
        assert dots[0] > dots[-1], (
            f"Adjacent positions should have higher dot product than distant ones: {dots}"
        )


# --------------------------------------------------------------------------- #
# RoPEMultiHeadAttention
# --------------------------------------------------------------------------- #

class TestRoPEMultiHeadAttention:
    def test_output_shape(self):
        attn = RoPEMultiHeadAttention(d_model=128, n_heads=4)
        x = torch.randn(2, 8, 128)  # [B, seq, d]
        out = attn(x)
        assert out.shape == (2, 8, 128)

    def test_backprop(self):
        attn = RoPEMultiHeadAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# --------------------------------------------------------------------------- #
# MetaEncoder
# --------------------------------------------------------------------------- #

class TestMetaEncoder:
    def test_output_is_list_of_L_tensors(self):
        enc = MetaEncoder(LAYER_DIMS, projection_dim=128)
        z_list = enc(make_traj())
        assert isinstance(z_list, list)
        assert len(z_list) == len(LAYER_DIMS)

    def test_each_output_is_unit_norm(self):
        enc = MetaEncoder(LAYER_DIMS, projection_dim=128)
        z_list = enc(make_traj())
        for z_l in z_list:
            norms = z_l.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_output_shapes(self):
        proj_dim = 128
        enc = MetaEncoder(LAYER_DIMS, projection_dim=proj_dim)
        z_list = enc(make_traj(B=8))
        for z_l in z_list:
            assert z_l.shape == (8, proj_dim)

    def test_different_batch_sizes(self):
        enc = MetaEncoder(LAYER_DIMS, projection_dim=64, n_heads=4)
        for B in [1, 4, 16]:
            z_list = enc(make_traj(B=B))
            assert all(z.shape[0] == B for z in z_list)

    def test_backprop(self):
        enc = MetaEncoder(LAYER_DIMS, projection_dim=128)
        traj = [torch.randn(4, d, requires_grad=True) for d in LAYER_DIMS]
        z_list = enc(traj)
        sum(z.sum() for z in z_list).backward()
        assert all(t.grad is not None for t in traj)

    def test_wrong_num_layers_raises(self):
        enc = MetaEncoder(LAYER_DIMS, projection_dim=128)
        with pytest.raises(AssertionError):
            enc([torch.randn(4, 64) for _ in range(3)])

    def test_projection_dim_must_divide_by_heads(self):
        with pytest.raises(AssertionError):
            MetaEncoder(LAYER_DIMS, projection_dim=130, n_heads=4)


# --------------------------------------------------------------------------- #
# ProfileRegressor
# --------------------------------------------------------------------------- #

class TestProfileRegressor:
    def test_output_shape(self):
        """output_dim controls the last dimension of the output tensor."""
        reg = ProfileRegressor(input_dim=128, hidden_dim=64, output_dim=256)
        z_product = torch.randn(16, 128)
        out = reg(z_product)
        assert out.shape == (16, 256)

    def test_output_shape_small(self):
        """Works with any output_dim including 1."""
        reg = ProfileRegressor(input_dim=64, hidden_dim=32, output_dim=16)
        out = reg(torch.randn(8, 64))
        assert out.shape == (8, 16)

    def test_backprop(self):
        reg = ProfileRegressor(input_dim=64, hidden_dim=32, output_dim=256)
        z = torch.randn(8, 64, requires_grad=True)
        out = reg(z)
        out.sum().backward()
        assert z.grad is not None

    def test_symmetric_input(self):
        """z_a * z_b should equal z_b * z_a, producing identical output."""
        reg = ProfileRegressor(input_dim=64, hidden_dim=32, output_dim=256)
        z_a = torch.randn(4, 64)
        z_b = torch.randn(4, 64)
        out_ab = reg(z_a * z_b)
        out_ba = reg(z_b * z_a)
        assert torch.allclose(out_ab, out_ba, atol=1e-6)
