"""Conditional VAE with classifier-free conditional dropout (Ho & Salimans 2022)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from model.models.cond_embed import ConditionEmbedder
from model.tokenizer import N_DAY, N_JOINT, N_YEAR_UNITS


class CVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 512,
        embed_dim: int = 32,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cfg_dropout = cfg_dropout
        self.cond_embed = ConditionEmbedder(embed_dim)
        cd = self.cond_embed.out_dim
        self.null_cond = nn.Parameter(torch.randn(cd) * 0.02)

        self.encoder = nn.Sequential(
            nn.Linear(cd + N_DAY + N_YEAR_UNITS, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cd, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.joint_head = nn.Linear(hidden, N_JOINT)

    def _cond_or_null(self, cond_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        c = self.cond_embed(cond_idx)
        return torch.where(mask.unsqueeze(-1), self.null_cond.expand_as(c), c)

    def encode(self, c: torch.Tensor, d_idx: torch.Tensor, yu_idx: torch.Tensor):
        d_oh = F.one_hot(d_idx, N_DAY).float()
        y_oh = F.one_hot(yu_idx, N_YEAR_UNITS).float()
        h = self.encoder(torch.cat([c, d_oh, y_oh], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.decoder(torch.cat([z, c], dim=-1))
        return self.joint_head(h)

    def forward(
        self,
        cond_idx: torch.Tensor,
        d_idx: torch.Tensor,
        yu_idx: torch.Tensor,
        training_cfg_dropout: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cond_idx.size(0)
        if training_cfg_dropout:
            mask = torch.rand(B, device=cond_idx.device) < self.cfg_dropout
        else:
            mask = torch.zeros(B, dtype=torch.bool, device=cond_idx.device)
        c = self._cond_or_null(cond_idx, mask)
        mu, logvar = self.encode(c, d_idx, yu_idx)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        logits = self.decode(z, c)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, cond_idx: torch.Tensor, w: float = 2.5) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cond_idx.size(0)
        device = cond_idx.device
        z = torch.randn(B, self.latent_dim, device=device)
        cond_emb = self._cond_or_null(cond_idx, torch.zeros(B, dtype=torch.bool, device=device))
        null_emb = self._cond_or_null(cond_idx, torch.ones(B, dtype=torch.bool, device=device))
        l_cond = self.decode(z, cond_emb)
        l_null = self.decode(z, null_emb)
        logits = w * l_cond + (1.0 - w) * l_null
        joint = dist.Categorical(logits=logits).sample()
        return joint // N_YEAR_UNITS, joint % N_YEAR_UNITS


def cvae_loss(
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    joint_idx: torch.Tensor,
    beta: float,
    free_bits: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.cross_entropy(logits, joint_idx)
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
    kl = kl_per_dim.sum()
    return recon + beta * kl, recon, kl
