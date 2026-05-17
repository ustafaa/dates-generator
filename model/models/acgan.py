"""Conditional GAN with hybrid MLE + WGAN-GP + projection-D.

References:
- Miyato & Koyama 2018 (projection discriminator)
- Gulrajani et al. 2017 (WGAN-GP)
- Che et al. 2017 (MaliGAN: MLE-augmented adversarial training for discrete data)
- Jang/Maddison 2016 (Gumbel-Softmax)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.cond_embed import ConditionEmbedder
from model.tokenizer import N_JOINT, N_YEAR_UNITS


def _sn(layer: nn.Module) -> nn.Module:
    return nn.utils.parametrizations.spectral_norm(layer)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 32, hidden: int = 256, embed_dim: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_embed = ConditionEmbedder(embed_dim)
        cd = self.cond_embed.out_dim
        self.trunk = nn.Sequential(
            nn.Linear(noise_dim + cd, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, N_JOINT)

    def forward(self, z, cond_idx, tau: float = 1.0, hard: bool = False):
        c = self.cond_embed(cond_idx)
        h = self.trunk(torch.cat([z, c], dim=-1))
        logits = self.head(h)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    def logits(self, z, cond_idx):
        c = self.cond_embed(cond_idx)
        h = self.trunk(torch.cat([z, c], dim=-1))
        return self.head(h)


class Discriminator(nn.Module):
    def __init__(self, hidden: int = 256, embed_dim: int = 32):
        super().__init__()
        self.cond_embed = ConditionEmbedder(embed_dim)
        cd = self.cond_embed.out_dim
        self.trunk = nn.Sequential(
            _sn(nn.Linear(N_JOINT, hidden)), nn.LeakyReLU(0.2),
            _sn(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2),
            _sn(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2),
        )
        self.rf = _sn(nn.Linear(hidden, 1))
        self.proj = _sn(nn.Linear(cd, hidden))

    def forward(self, x_soft, cond_idx):
        phi = self.trunk(x_soft)
        c_emb = self.cond_embed(cond_idx)
        rf = self.rf(phi).squeeze(-1)
        proj = (self.proj(c_emb) * phi).sum(-1)
        return rf + proj


def gradient_penalty(D: Discriminator, x_real, x_fake, cond_idx):
    B = x_real.size(0)
    alpha = torch.rand(B, 1, device=x_real.device)
    x_interp = (alpha * x_real + (1.0 - alpha) * x_fake).detach().requires_grad_(True)
    out = D(x_interp, cond_idx)
    grads = torch.autograd.grad(
        outputs=out.sum(), inputs=x_interp,
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=-1) - 1.0) ** 2).mean()


def acgan_g_step_loss(
    G: Generator,
    D: Discriminator,
    cond_idx,
    joint_idx_real,
    tau: float,
    lambda_mle: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = cond_idx.size(0)
    z = torch.randn(B, G.noise_dim, device=cond_idx.device)
    x_fake = G(z, cond_idx, tau=tau, hard=False)
    adv = -D(x_fake, cond_idx).mean()
    mle = F.cross_entropy(G.logits(z, cond_idx), joint_idx_real)
    loss = adv + lambda_mle * mle
    return loss, {"g_adv": float(adv.detach()), "g_mle": float(mle.detach())}


def acgan_d_step_loss(
    G: Generator,
    D: Discriminator,
    cond_idx,
    joint_idx_real,
    tau: float,
    lambda_gp: float = 10.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = cond_idx.size(0)
    x_real = F.one_hot(joint_idx_real, N_JOINT).float()
    z = torch.randn(B, G.noise_dim, device=cond_idx.device)
    with torch.no_grad():
        x_fake = G(z, cond_idx, tau=tau, hard=False)
    d_real = D(x_real, cond_idx).mean()
    d_fake = D(x_fake, cond_idx).mean()
    wasserstein = d_fake - d_real
    gp = gradient_penalty(D, x_real, x_fake, cond_idx)
    loss = wasserstein + lambda_gp * gp
    return loss, {"d_real": float(d_real.detach()), "d_fake": float(d_fake.detach()), "gp": float(gp.detach())}


@torch.no_grad()
def acgan_sample(G: Generator, cond_idx, tau: float = 0.3):
    z = torch.randn(cond_idx.size(0), G.noise_dim, device=cond_idx.device)
    soft = G(z, cond_idx, tau=tau, hard=True)
    s = soft.argmax(-1)
    return s // N_YEAR_UNITS, s % N_YEAR_UNITS
