"""
Unit tests for InfoLoss.
Run with: pytest tests/
"""

import torch

from losses.info_loss import InfoLoss


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

D = 64       # projection dim (z vector dim)
L = 8        # layers
B = 16       # batch size
LAYER_DIMS = [16] * L  # small D_flow per layer for testing (prod uses 256)


def make_z_list(B=B, L=L, d=D):
    """Random L2-normalized per-layer z-vectors."""
    z_list = []
    for _ in range(L):
        z = torch.randn(B, d)
        z = z / z.norm(dim=-1, keepdim=True)
        z_list.append(z)
    return z_list


def make_rich_targets(N_pairs=100, layer_dims=LAYER_DIMS):
    """Random flow co-activation vectors for pairs (one per layer)."""
    return [torch.rand(N_pairs, D_l) for D_l in layer_dims]


def make_true_sims_matrix(B=B, L=L):
    """Random pairwise similarity matrix [B, B, L]."""
    sims = torch.rand(B, B, L)
    # Make symmetric
    sims = (sims + sims.transpose(0, 1)) / 2
    return sims


# --------------------------------------------------------------------------- #
# InfoLoss
# --------------------------------------------------------------------------- #

class TestInfoLoss:
    def _make_loss(self, layer_dims=LAYER_DIMS):
        return InfoLoss(layer_dims=layer_dims, projection_dim=D, hidden_dim=32)

    def test_output_is_scalar(self):
        loss_fn = self._make_loss()
        N = 50
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]
        targets = make_rich_targets(N)
        loss = loss_fn(z_a, z_b, targets)
        assert loss.dim() == 0

    def test_positive_loss(self):
        loss_fn = self._make_loss()
        N = 50
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]
        targets = make_rich_targets(N)
        loss = loss_fn(z_a, z_b, targets)
        assert loss.item() > 0

    def test_perfect_prediction_gives_low_loss(self):
        """If each regressor perfectly predicts its rich target, loss should be ~0."""
        loss_fn = self._make_loss()
        N = 20
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]

        # Compute what each regressor would predict for these z-products
        with torch.no_grad():
            fake_targets = [
                loss_fn.regressors[l](z_a[l] * z_b[l])
                for l in range(L)
            ]

        loss = loss_fn(z_a, z_b, fake_targets)
        assert loss.item() < 0.01

    def test_backprop(self):
        loss_fn = self._make_loss()
        N = 20
        z_a = [torch.randn(N, D, requires_grad=True) for _ in range(L)]
        z_b = [torch.randn(N, D, requires_grad=True) for _ in range(L)]
        targets = make_rich_targets(N)
        loss = loss_fn(z_a, z_b, targets)
        loss.backward()
        assert all(z.grad is not None for z in z_a)

    def test_regressor_output_shape(self):
        """Each regressor should output [N, D_l] for its layer."""
        loss_fn = self._make_loss()
        N = 30
        for l, D_l in enumerate(LAYER_DIMS):
            z_prod = torch.randn(N, D)
            out = loss_fn.regressors[l](z_prod)
            assert out.shape == (N, D_l), f"Layer {l}: expected ({N}, {D_l}), got {out.shape}"

    def test_varying_layer_dims(self):
        """InfoLoss handles layers with different D_l values."""
        layer_dims = [8, 16, 32, 64, 32, 16, 8, 4]
        loss_fn = InfoLoss(layer_dims=layer_dims, projection_dim=D, hidden_dim=32)
        N = 20
        z_a = [torch.randn(N, D) for _ in range(L)]
        z_b = [torch.randn(N, D) for _ in range(L)]
        targets = [torch.rand(N, D_l) for D_l in layer_dims]
        loss = loss_fn(z_a, z_b, targets)
        assert loss.dim() == 0
        assert loss.item() > 0
