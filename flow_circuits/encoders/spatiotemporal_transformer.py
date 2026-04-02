from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryDepthEmbedding(nn.Module):
    def __init__(self, dim: int, max_layers: int = 64) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_layers).float()
        angles = positions[:, None] * freqs[None, :]
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[layer_positions].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[layer_positions].unsqueeze(0).unsqueeze(0)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class CausalDepthAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rope = RotaryDepthEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        layer_positions: torch.Tensor,
        attn_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, layer_positions)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~attn_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        if return_attention:
            return output, weights.mean(dim=1)
        return output, None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalDepthAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_positions: torch.Tensor,
        attn_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attention = self.attn(
            self.norm1(x),
            layer_positions=layer_positions,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x, attention


class SpatiotemporalEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        grid_size: int = 4,
        token_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.grid_size = grid_size
        self.n_cells = grid_size * grid_size
        self.token_dim = token_dim
        mlp_dim = mlp_dim or token_dim * 2
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=token_dim,
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(n_transformer_layers)
        )
        self.final_norm = nn.LayerNorm(token_dim)
        layer_positions = torch.arange(n_layers).repeat_interleave(self.n_cells)
        self.register_buffer("layer_positions", layer_positions, persistent=False)
        mask = layer_positions.view(-1, 1) >= layer_positions.view(1, -1)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(
        self,
        token_inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = token_inputs.shape[0]
        seq = token_inputs.view(batch_size, self.n_layers * self.n_cells, self.token_dim)
        attention = None
        for block_idx, block in enumerate(self.blocks):
            seq, layer_attention = block(
                seq,
                layer_positions=self.layer_positions,
                attn_mask=self.attn_mask,
                return_attention=return_attention and block_idx == len(self.blocks) - 1,
            )
            if layer_attention is not None:
                attention = layer_attention
        seq = self.final_norm(seq)
        seq = F.normalize(seq, dim=-1)
        z = seq.view(batch_size, self.n_layers, self.n_cells, self.token_dim)
        return z, attention
