"""Transformer building blocks used by FluxVacuumTransformer."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimestepMLP(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.net = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.model_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half, 1)
        )
        phase = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
        if embedding.shape[-1] < self.model_dim:
            embedding = F.pad(embedding, (0, self.model_dim - embedding.shape[-1]))
        return self.net(embedding)


class ContinuousTimeMLP(nn.Module):
    """Embed a scalar mask probability directly into the AdaLN condition."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, int(model_dim)),
            nn.SiLU(),
            nn.Linear(int(model_dim), int(model_dim)),
        )

    def forward(self, mask_probability: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(mask_probability).float().reshape(-1)
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0.0) | (value > 1.0)).any()
        ):
            raise ValueError("Continuous time conditioning must lie in [0, 1]")
        return self.net(value.unsqueeze(-1))


class SelfAttention(nn.Module):
    """Masked self-attention without rotary or sinusoidal transforms."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(model_dim) // self.num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = float(dropout)

    def forward(self, hidden: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        batch, length, model_dim = hidden.shape
        qkv = self.qkv(hidden).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~active_mask.bool().view(batch, 1, 1, length),
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, model_dim)
        return self.output(attended) * active_mask.unsqueeze(-1).to(attended.dtype)


def _modulate(
    hidden: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return hidden * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNTransformerBlock(nn.Module):
    """AdaLN-Zero block whose global condition is time plus geometry."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attention = SelfAttention(model_dim, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * int(mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * int(mlp_ratio), model_dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 6 * model_dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        modulation = self.adaln(condition)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(
            6, dim=-1
        )
        hidden = hidden + gate_a.unsqueeze(1) * self.attention(
            _modulate(self.norm_attention(hidden), shift_a, scale_a), active_mask
        )
        hidden = hidden + gate_m.unsqueeze(1) * self.mlp(
            _modulate(self.norm_mlp(hidden), shift_m, scale_m)
        )
        return hidden * active_mask.unsqueeze(-1).to(hidden.dtype)


class QueryCrossAttentionBlock(nn.Module):
    """Cross-attention and MLP block without query-to-query information leaks."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.memory_norm = nn.LayerNorm(model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * int(mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * int(mlp_ratio), model_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        update, _ = self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask.bool(),
            need_weights=False,
        )
        queries = queries + update
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries * query_mask.unsqueeze(-1).to(queries.dtype)
