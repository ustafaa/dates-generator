"""Shared 4-table condition embedder for all four models."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.tokenizer import N_DEC, N_DOW, N_LEAP, N_MON


class ConditionEmbedder(nn.Module):
    """(DOW, MON, LEAP, DEC) indices -> 4*embed_dim vector by concat."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.dow = nn.Embedding(N_DOW, embed_dim)
        self.mon = nn.Embedding(N_MON, embed_dim)
        self.leap = nn.Embedding(N_LEAP, embed_dim)
        self.dec = nn.Embedding(N_DEC, embed_dim)
        self.out_dim = 4 * embed_dim

    def forward(self, cond_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.dow(cond_idx[:, 0]),
                self.mon(cond_idx[:, 1]),
                self.leap(cond_idx[:, 2]),
                self.dec(cond_idx[:, 3]),
            ],
            dim=-1,
        )
