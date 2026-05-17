"""MaskGIT (Chang et al. CVPR 2022) -- non-autoregressive masked generative
transformer with cosine masking schedule, classifier-free guidance, and
confidence-based iterative parallel decoding."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from model.models.cond_embed import ConditionEmbedder
from model.tokenizer import N_DAY, N_YEAR_UNITS


class MaskGIT(nn.Module):
    DAY_MASK = N_DAY
    YU_MASK = N_YEAR_UNITS

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        embed_dim: int = 32,
        cfg_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg_dropout = cfg_dropout
        self.cond_embed = ConditionEmbedder(embed_dim)
        self.cond_proj = nn.Linear(self.cond_embed.out_dim, d_model)
        self.null_cond = nn.Parameter(torch.randn(d_model) * 0.02)
        self.day_embed = nn.Embedding(N_DAY + 1, d_model)
        self.yu_embed = nn.Embedding(N_YEAR_UNITS + 1, d_model)
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.day_head = nn.Linear(d_model, N_DAY)
        self.yu_head = nn.Linear(d_model, N_YEAR_UNITS)

    def forward(self, d_in, yu_in, cond_idx, cfg_mask=None):
        B = d_in.size(0)
        device = d_in.device
        if cfg_mask is None:
            cfg_mask = torch.zeros(B, dtype=torch.bool, device=device)
        c_proj = self.cond_proj(self.cond_embed(cond_idx))
        c_tok = torch.where(cfg_mask.unsqueeze(-1), self.null_cond.expand_as(c_proj), c_proj)
        d_tok = self.day_embed(d_in)
        yu_tok = self.yu_embed(yu_in)
        seq = torch.stack([c_tok, d_tok, yu_tok], dim=1) + self.pos.unsqueeze(0)
        h = self.backbone(seq)
        return self.day_head(h[:, 1]), self.yu_head(h[:, 2])

    def _cfg_forward(self, d_in, yu_in, cond_idx, w):
        B = d_in.size(0)
        device = d_in.device
        d_cond, yu_cond = self.forward(d_in, yu_in, cond_idx,
                                       cfg_mask=torch.zeros(B, dtype=torch.bool, device=device))
        d_null, yu_null = self.forward(d_in, yu_in, cond_idx,
                                       cfg_mask=torch.ones(B, dtype=torch.bool, device=device))
        return w * d_cond + (1.0 - w) * d_null, w * yu_cond + (1.0 - w) * yu_null

    @torch.no_grad()
    def sample(self, cond_idx, w: float = 2.5):
        B = cond_idx.size(0)
        device = cond_idx.device
        d_idx = torch.full((B,), self.DAY_MASK, dtype=torch.long, device=device)
        yu_idx = torch.full((B,), self.YU_MASK, dtype=torch.long, device=device)
        d_l, yu_l = self._cfg_forward(d_idx, yu_idx, cond_idx, w)
        d_probs = F.softmax(d_l, dim=-1)
        yu_probs = F.softmax(yu_l, dim=-1)
        d_samp = dist.Categorical(probs=d_probs).sample()
        yu_samp = dist.Categorical(probs=yu_probs).sample()
        d_conf = d_probs.gather(-1, d_samp.unsqueeze(-1)).squeeze(-1)
        yu_conf = yu_probs.gather(-1, yu_samp.unsqueeze(-1)).squeeze(-1)
        commit_d = d_conf >= yu_conf
        d_idx = torch.where(commit_d, d_samp, d_idx)
        yu_idx = torch.where(~commit_d, yu_samp, yu_idx)
        d_l, yu_l = self._cfg_forward(d_idx, yu_idx, cond_idx, w)
        d_samp = dist.Categorical(logits=d_l).sample()
        yu_samp = dist.Categorical(logits=yu_l).sample()
        d_idx = torch.where(commit_d, d_idx, d_samp)
        yu_idx = torch.where(commit_d, yu_samp, yu_idx)
        return d_idx, yu_idx


def maskgit_loss(model, cond_idx, d_idx, yu_idx):
    B = cond_idx.size(0)
    device = cond_idx.device
    t = torch.rand(B, device=device)
    mask_rate = torch.cos(math.pi * t / 2.0)
    mask_d = torch.rand(B, device=device) < mask_rate
    mask_yu = torch.rand(B, device=device) < mask_rate
    no_mask = ~(mask_d | mask_yu)
    coin = torch.rand(B, device=device) < 0.5
    mask_d = mask_d | (no_mask & coin)
    mask_yu = mask_yu | (no_mask & ~coin)
    d_in = torch.where(mask_d, torch.full_like(d_idx, MaskGIT.DAY_MASK), d_idx)
    yu_in = torch.where(mask_yu, torch.full_like(yu_idx, MaskGIT.YU_MASK), yu_idx)
    cfg_mask = torch.rand(B, device=device) < model.cfg_dropout
    d_logits, yu_logits = model(d_in, yu_in, cond_idx, cfg_mask=cfg_mask)
    ce_d = F.cross_entropy(d_logits, d_idx, reduction="none") * mask_d.float()
    ce_yu = F.cross_entropy(yu_logits, yu_idx, reduction="none") * mask_yu.float()
    return (
        ce_d.sum() / mask_d.float().sum().clamp(min=1.0)
        + ce_yu.sum() / mask_yu.float().sum().clamp(min=1.0)
    )
