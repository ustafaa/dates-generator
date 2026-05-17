"""Shared training utilities and per-model trainers."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from model.data import DateDataset, Record
from model.metrics import csr_report
from model.models.acgan import (
    Discriminator,
    Generator,
    acgan_d_step_loss,
    acgan_g_step_loss,
    acgan_sample,
)
from model.models.cvae import CVAE, cvae_loss
from model.models.maskgit import MaskGIT, maskgit_loss
from model.models.mdlm import MDLM, mdlm_loss
from model.tokenizer import N_YEAR_UNITS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def _eval_csr(generate_fn: Callable, records: List[Record], max_n: int = 2000):
    if max_n is not None and len(records) > max_n:
        records = records[:max_n]
    conds = [r[0] for r in records]
    out = []
    for i in range(0, len(conds), 1024):
        chunk = conds[i : i + 1024]
        out.extend(generate_fn(chunk))
    gens = [(c, d, m, y) for c, (d, m, y) in zip(conds, out)]
    return csr_report(gens)


def _conds_to_idx(conds, device):
    rows = [list(c.as_indices()) for c in conds]
    return torch.tensor(rows, dtype=torch.long, device=device)


# ---------------------------- CVAE ----------------------------
def train_cvae(
    train_recs, val_recs,
    epochs: int = 15, batch_size: int = 1024, lr: float = 1e-3,
    latent_dim: int = 32, hidden: int = 512,
    beta_warmup_epochs: int = 3, free_bits: float = 0.02,
    cfg_dropout: float = 0.1, sample_w: float = 2.5,
    val_eval_n: int = 2000, verbose: bool = True, seed: int = 0,
):
    torch.manual_seed(seed)
    device = _device()
    ds = DateDataset(train_recs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = CVAE(latent_dim=latent_dim, hidden=hidden, cfg_dropout=cfg_dropout).to(device)
    if verbose:
        print(f"[cvae] params: {_count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist: List[Dict] = []

    def gen_fn(conds):
        idx = _conds_to_idx(conds, device)
        d, yu = model.sample(idx, w=sample_w)
        return [(int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item()))
                for i, c in enumerate(conds)]

    for epoch in range(epochs):
        model.train()
        beta = min(1.0, (epoch + 1) / max(1, beta_warmup_epochs))
        t0 = time.time()
        rl = rr = rk = 0.0
        nb = 0
        for cond, d_idx, yu_idx in loader:
            cond = cond.to(device); d_idx = d_idx.to(device); yu_idx = yu_idx.to(device)
            logits, mu, logvar = model(cond, d_idx, yu_idx)
            joint = d_idx * N_YEAR_UNITS + yu_idx
            loss, recon, kl = cvae_loss(logits, mu, logvar, joint, beta, free_bits)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rl += loss.item(); rr += recon.item(); rk += kl.item(); nb += 1
        model.train(False)
        rep = _eval_csr(gen_fn, val_recs, max_n=val_eval_n)
        hist.append({"epoch": epoch, "loss": rl/nb, "recon": rr/nb, "kl": rk/nb,
                    "beta": beta, "val_csr_all": rep.csr_all,
                    "val_csr_dow": rep.csr_dow, "val_csr_leap": rep.csr_leap,
                    "val_valid": rep.valid_rate, "time": time.time() - t0})
        if verbose:
            print(f"[cvae] ep{epoch:>2d} loss={rl/nb:.4f} val_CSR={rep.csr_all:.3f}")
    return model, hist


# ---------------------------- AC-GAN ----------------------------
def train_acgan(
    train_recs, val_recs,
    epochs: int = 25, batch_size: int = 1024, lr: float = 2e-4,
    noise_dim: int = 32, hidden: int = 256,
    tau_start: float = 1.0, tau_end: float = 0.3,
    lambda_mle: float = 0.5, lambda_gp: float = 10.0,
    d_steps_per_g: int = 1,
    val_eval_n: int = 1500, verbose: bool = True, seed: int = 0,
):
    torch.manual_seed(seed)
    device = _device()
    ds = DateDataset(train_recs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    G = Generator(noise_dim=noise_dim, hidden=hidden).to(device)
    D = Discriminator(hidden=hidden).to(device)
    if verbose:
        print(f"[acgan] G={_count_params(G):,} D={_count_params(D):,}")
    optG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.5, 0.9))
    hist: List[Dict] = []

    def gen_fn(conds):
        idx = _conds_to_idx(conds, device)
        d, yu = acgan_sample(G, idx, tau=tau_end)
        return [(int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item()))
                for i, c in enumerate(conds)]

    for epoch in range(epochs):
        G.train(); D.train()
        tau = tau_start + (tau_end - tau_start) * (epoch / max(1, epochs - 1))
        t0 = time.time(); rl_d = rl_g = 0.0; nb = 0
        for cond, d_idx, yu_idx in loader:
            cond = cond.to(device); d_idx = d_idx.to(device); yu_idx = yu_idx.to(device)
            joint = d_idx * N_YEAR_UNITS + yu_idx
            for _ in range(d_steps_per_g):
                d_loss, _ = acgan_d_step_loss(G, D, cond, joint, tau=tau, lambda_gp=lambda_gp)
                optD.zero_grad(); d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                optD.step()
            g_loss, _ = acgan_g_step_loss(G, D, cond, joint, tau=tau, lambda_mle=lambda_mle)
            optG.zero_grad(); g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            optG.step()
            rl_d += d_loss.item(); rl_g += g_loss.item(); nb += 1
        G.train(False)
        rep = _eval_csr(gen_fn, val_recs, max_n=val_eval_n)
        hist.append({"epoch": epoch, "d_loss": rl_d/nb, "g_loss": rl_g/nb, "tau": tau,
                    "val_csr_all": rep.csr_all, "val_csr_dow": rep.csr_dow,
                    "val_csr_leap": rep.csr_leap, "val_valid": rep.valid_rate,
                    "time": time.time() - t0})
        if verbose:
            print(f"[acgan] ep{epoch:>2d} d={rl_d/nb:.3f} g={rl_g/nb:.3f} tau={tau:.2f} val_CSR={rep.csr_all:.3f}")
    return G, D, hist


# ---------------------------- MaskGIT ----------------------------
def train_maskgit(
    train_recs, val_recs,
    epochs: int = 15, batch_size: int = 1024, lr: float = 3e-3,
    d_model: int = 128, n_layers: int = 2, n_heads: int = 4, ff_dim: int = 256,
    cfg_dropout: float = 0.1, sample_w: float = 2.5,
    val_eval_n: int = 2000, verbose: bool = True, seed: int = 0,
):
    torch.manual_seed(seed)
    device = _device()
    ds = DateDataset(train_recs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MaskGIT(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    ff_dim=ff_dim, cfg_dropout=cfg_dropout).to(device)
    if verbose:
        print(f"[maskgit] params: {_count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist: List[Dict] = []

    def gen_fn(conds):
        idx = _conds_to_idx(conds, device)
        d, yu = model.sample(idx, w=sample_w)
        return [(int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item()))
                for i, c in enumerate(conds)]

    for epoch in range(epochs):
        model.train()
        t0 = time.time(); rl = 0.0; nb = 0
        for cond, d_idx, yu_idx in loader:
            cond = cond.to(device); d_idx = d_idx.to(device); yu_idx = yu_idx.to(device)
            loss = maskgit_loss(model, cond, d_idx, yu_idx)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rl += loss.item(); nb += 1
        model.train(False)
        rep = _eval_csr(gen_fn, val_recs, max_n=val_eval_n)
        hist.append({"epoch": epoch, "loss": rl/nb, "val_csr_all": rep.csr_all,
                    "val_csr_dow": rep.csr_dow, "val_csr_leap": rep.csr_leap,
                    "val_valid": rep.valid_rate, "time": time.time() - t0})
        if verbose:
            print(f"[maskgit] ep{epoch:>2d} loss={rl/nb:.4f} val_CSR={rep.csr_all:.3f}")
    return model, hist


# ---------------------------- MDLM ----------------------------
def train_mdlm(
    train_recs, val_recs,
    epochs: int = 15, batch_size: int = 1024, lr: float = 1e-3,
    hidden: int = 512, T: int = 20,
    cfg_dropout: float = 0.1, sample_w: float = 2.5,
    val_eval_n: int = 2000, verbose: bool = True, seed: int = 0,
):
    torch.manual_seed(seed)
    device = _device()
    ds = DateDataset(train_recs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MDLM(hidden=hidden, T=T, cfg_dropout=cfg_dropout).to(device)
    if verbose:
        print(f"[mdlm] params: {_count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist: List[Dict] = []

    def gen_fn(conds):
        idx = _conds_to_idx(conds, device)
        d, yu = model.sample(idx, w=sample_w)
        return [(int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item()))
                for i, c in enumerate(conds)]

    for epoch in range(epochs):
        model.train()
        t0 = time.time(); rl = 0.0; nb = 0
        for cond, d_idx, yu_idx in loader:
            cond = cond.to(device); d_idx = d_idx.to(device); yu_idx = yu_idx.to(device)
            joint = d_idx * N_YEAR_UNITS + yu_idx
            loss = mdlm_loss(model, cond, joint)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            rl += loss.item(); nb += 1
        model.train(False)
        rep = _eval_csr(gen_fn, val_recs, max_n=val_eval_n)
        hist.append({"epoch": epoch, "loss": rl/nb, "val_csr_all": rep.csr_all,
                    "val_csr_dow": rep.csr_dow, "val_csr_leap": rep.csr_leap,
                    "val_valid": rep.valid_rate, "time": time.time() - t0})
        if verbose:
            print(f"[mdlm] ep{epoch:>2d} loss={rl/nb:.4f} val_CSR={rep.csr_all:.3f}")
    return model, hist
