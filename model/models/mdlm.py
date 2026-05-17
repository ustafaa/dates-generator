"""MDLM (Sahoo et al. NeurIPS 2024) -- Masked Diffusion Language Model with
classifier-free guidance and a joint 310-way head.

Single-token framing: the output is one categorical token over
{0..309, MASK=310}. Absorbing-state forward process with linear alpha_t = 1 - t.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from model.models.cond_embed import ConditionEmbedder
from model.tokenizer import N_JOINT, N_YEAR_UNITS


class MDLM(nn.Module):
    MASK_ID = N_JOINT  # 310

    def __init__(
        self,
        hidden: int = 512,
        embed_dim: int = 32,
        T: int = 20,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.cfg_dropout = cfg_dropout
        self.cond_embed = ConditionEmbedder(embed_dim)
        cd = self.cond_embed.out_dim
        self.null_cond = nn.Parameter(torch.randn(cd) * 0.02)
        self.x_embed = nn.Embedding(N_JOINT + 1, embed_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        in_dim = cd + embed_dim + embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.joint_head = nn.Linear(hidden, N_JOINT)

    def _cond(self, cond_idx, cfg_mask):
        c = self.cond_embed(cond_idx)
        return torch.where(cfg_mask.unsqueeze(-1), self.null_cond.expand_as(c), c)

    def forward(self, x_in, cond_idx, t, cfg_mask=None):
        B = x_in.size(0)
        if cfg_mask is None:
            cfg_mask = torch.zeros(B, dtype=torch.bool, device=x_in.device)
        c = self._cond(cond_idx, cfg_mask)
        x_emb = self.x_embed(x_in)
        t_emb = self.t_embed(t.unsqueeze(-1))
        h = self.trunk(torch.cat([c, x_emb, t_emb], dim=-1))
        return self.joint_head(h)

    def _cfg_logits(self, x_in, cond_idx, t, w):
        B = x_in.size(0)
        device = x_in.device
        l_cond = self.forward(x_in, cond_idx, t,
                              cfg_mask=torch.zeros(B, dtype=torch.bool, device=device))
        l_null = self.forward(x_in, cond_idx, t,
                              cfg_mask=torch.ones(B, dtype=torch.bool, device=device))
        return w * l_cond + (1.0 - w) * l_null

    @torch.no_grad()
    def sample(self, cond_idx, w: float = 2.5):
        B = cond_idx.size(0)
        device = cond_idx.device
        x = torch.full((B,), self.MASK_ID, dtype=torch.long, device=device)
        for step in range(self.T, 0, -1):
            t = torch.full((B,), step / self.T, device=device)
            logits = self._cfg_logits(x, cond_idx, t, w)
            x_samp = dist.Categorical(logits=logits).sample()
            masked = x == self.MASK_ID
            unmask = (torch.rand(B, device=device) < 1.0 / step) & masked
            x = torch.where(unmask, x_samp, x)
        if (x == self.MASK_ID).any():
            t = torch.full((B,), 1.0 / self.T, device=device)
            logits = self._cfg_logits(x, cond_idx, t, w)
            x_samp = dist.Categorical(logits=logits).sample()
            x = torch.where(x == self.MASK_ID, x_samp, x)
        return x // N_YEAR_UNITS, x % N_YEAR_UNITS


def mdlm_loss(model, cond_idx, joint_idx):
    B = cond_idx.size(0)
    device = cond_idx.device
    t = torch.rand(B, device=device).clamp(min=1e-3, max=1.0 - 1e-3)
    mask = torch.rand(B, device=device) < t
    x_in = torch.where(mask, torch.full_like(joint_idx, MDLM.MASK_ID), joint_idx)
    cfg_mask = torch.rand(B, device=device) < model.cfg_dropout
    logits = model(x_in, cond_idx, t, cfg_mask=cfg_mask)
    weight = 1.0 / t
    ce = F.cross_entropy(logits, joint_idx, reduction="none")
    return (weight * ce * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
