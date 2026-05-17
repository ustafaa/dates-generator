#!/usr/bin/env python

from __future__ import annotations
import argparse, math, random, re, calendar
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Tokenizer (inlined) ----------------
DOW_TOKENS = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
MON_TOKENS = [
    "[JAN]",
    "[FEB]",
    "[MAR]",
    "[APR]",
    "[MAY]",
    "[JUN]",
    "[JUL]",
    "[AUG]",
    "[SEP]",
    "[OCT]",
    "[NOV]",
    "[DEC]",
]
LEAP_TOKENS = ["[False]", "[True]"]
DEC_TOKENS = [f"[{d}]" for d in range(180, 221)]
DOW_TO_IDX = {t: i for i, t in enumerate(DOW_TOKENS)}
MON_TO_IDX = {t: i for i, t in enumerate(MON_TOKENS)}
LEAP_TO_IDX = {t: i for i, t in enumerate(LEAP_TOKENS)}
DEC_TO_IDX = {t: i for i, t in enumerate(DEC_TOKENS)}
MON_NAME_TO_NUM = {t: i + 1 for i, t in enumerate(MON_TOKENS)}
N_DOW, N_MON, N_LEAP, N_DEC = 7, 12, 2, 41
N_YEAR_UNITS, N_DAY = 10, 31
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<mask>"]
DIGIT_TOKENS = [str(d) for d in range(10)]
VOCAB = SPECIALS + DOW_TOKENS + MON_TOKENS + LEAP_TOKENS + DEC_TOKENS + DIGIT_TOKENS
TOKEN_TO_ID = {t: i for i, t in enumerate(VOCAB)}
ID_TO_TOKEN = {i: t for t, i in TOKEN_TO_ID.items()}
PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
SEP_ID = TOKEN_TO_ID["<sep>"]
MASK_ID = TOKEN_TO_ID["<mask>"]
DIGIT_IDS = [TOKEN_TO_ID[str(d)] for d in range(10)]
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 10

CONDITION_RE = re.compile(
    r"^\[(?P<dow>[A-Z]{3})\]\s+\[(?P<mon>[A-Z]{3})\]\s+\[(?P<leap>True|False)\]\s+\[(?P<dec>\d{3})\]"
)


@dataclass(frozen=True)
class Condition:
    dow: str
    mon: str
    leap: str
    dec: str

    def as_indices(self):
        return (
            DOW_TO_IDX[self.dow],
            MON_TO_IDX[self.mon],
            LEAP_TO_IDX[self.leap],
            DEC_TO_IDX[self.dec],
        )

    @property
    def month_num(self):
        return MON_NAME_TO_NUM[self.mon]

    @property
    def decade_int(self):
        return int(self.dec.strip("[]"))

    def as_prefix(self):
        return f"{self.dow} {self.mon} {self.leap} {self.dec}"


def parse_condition_line(line):
    m = CONDITION_RE.match(line.strip())
    if not m:
        raise ValueError(f"Malformed: {line!r}")
    return Condition(
        f"[{m['dow']}]", f"[{m['mon']}]", f"[{m['leap']}]", f"[{m['dec']}]"
    )


def format_output_line(c, d, mo, y):
    return f"{c.as_prefix()} {d}-{mo}-{y}"


def valid_date(d, m, y):
    if not (1 <= m <= 12):
        return False
    return 1 <= d <= calendar.monthrange(y, m)[1]


# ---------------- Models (inlined) ----------------
class ConditionEmbedder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.dow = nn.Embedding(N_DOW, embed_dim)
        self.mon = nn.Embedding(N_MON, embed_dim)
        self.leap = nn.Embedding(N_LEAP, embed_dim)
        self.dec = nn.Embedding(N_DEC, embed_dim)
        self.out_dim = 4 * embed_dim

    def forward(self, c):
        return torch.cat(
            [
                self.dow(c[:, 0]),
                self.mon(c[:, 1]),
                self.leap(c[:, 2]),
                self.dec(c[:, 3]),
            ],
            dim=-1,
        )


class CVAE(nn.Module):
    def __init__(self, latent_dim=32, hidden=512, embed_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_embed = ConditionEmbedder(embed_dim)
        cd = self.cond_embed.out_dim
        self.encoder = nn.Sequential(
            nn.Linear(cd + N_DAY + N_YEAR_UNITS, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent_dim + cd, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.joint_head = nn.Linear(hidden, N_DAY * N_YEAR_UNITS)

    def decode(self, z, c):
        ce = self.cond_embed(c)
        h = self.decoder_trunk(torch.cat([z, ce], dim=-1))
        return self.joint_head(h)

    @torch.no_grad()
    def sample(self, c):
        z = torch.randn(c.size(0), self.latent_dim, device=c.device)
        jl = self.decode(z, c)
        ji = torch.distributions.Categorical(logits=jl).sample()
        return ji // N_YEAR_UNITS, ji % N_YEAR_UNITS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ARTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, VOCAB_SIZE)
        mask = torch.triu(torch.ones(SEQ_LEN - 1, SEQ_LEN - 1), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, ids):
        h = self.pos(self.embed(ids))
        h = self.backbone(h, mask=self.causal_mask, is_causal=True)
        return self.head(h)


def sn(m):
    return nn.utils.parametrizations.spectral_norm(m)


class GeneratorNet(nn.Module):
    def __init__(self, noise_dim=32, hidden=256, embed_dim=32):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_embed = ConditionEmbedder(embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(noise_dim + self.cond_embed.out_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, N_DAY * N_YEAR_UNITS)

    def forward(self, z, c, tau=1.0, hard=False):
        ce = self.cond_embed(c)
        h = self.trunk(torch.cat([z, ce], dim=-1))
        return F.gumbel_softmax(self.head(h), tau=tau, hard=hard, dim=-1)


class DiffusionNet(nn.Module):
    def __init__(self, hidden=512, embed_dim=32, T=10):
        super().__init__()
        self.T = T
        self.cond_embed = ConditionEmbedder(embed_dim)
        self.day_embed = nn.Embedding(N_DAY + 1, embed_dim)
        self.yu_embed = nn.Embedding(N_YEAR_UNITS + 1, embed_dim)
        self.t_embed = nn.Embedding(T + 1, embed_dim)
        in_dim = self.cond_embed.out_dim + 2 * embed_dim + embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.day_head = nn.Linear(hidden, N_DAY)
        self.yu_head = nn.Linear(hidden, N_YEAR_UNITS)

    def forward(self, d, y, c, t):
        ce = self.cond_embed(c)
        de = self.day_embed(d)
        ye = self.yu_embed(y)
        te = self.t_embed(t)
        h = self.trunk(torch.cat([ce, de, ye, te], dim=-1))
        return self.day_head(h), self.yu_head(h)


# ---------------- Inference helpers ----------------
def cond_to_idx_t(c, device):
    return torch.tensor([list(c.as_indices())], dtype=torch.long, device=device)


@torch.no_grad()
def gen_cvae(model, c, device):
    idx = cond_to_idx_t(c, device)
    d, yu = model.sample(idx)
    return d.item() + 1, c.month_num, c.decade_int * 10 + yu.item()


@torch.no_grad()
def gen_ar(model, c, device, temperature=1.0, top_p=0.9, max_tries=5):
    digit_t = torch.tensor(DIGIT_IDS, dtype=torch.long, device=device)

    def _one():
        ids = torch.full((1, SEQ_LEN - 1), PAD_ID, dtype=torch.long, device=device)
        ids[0, 0] = BOS_ID
        ids[0, 1] = TOKEN_TO_ID[c.dow]
        ids[0, 2] = TOKEN_TO_ID[c.mon]
        ids[0, 3] = TOKEN_TO_ID[c.leap]
        ids[0, 4] = TOKEN_TO_ID[c.dec]
        ids[0, 5] = SEP_ID
        for w in (6, 7, 8):
            logits = model(ids)[:, w - 1, :]
            digit_logits = logits[:, DIGIT_IDS]
            if temperature != 1.0:
                digit_logits = digit_logits / temperature
            probs = F.softmax(digit_logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cu = torch.cumsum(sp, dim=-1)
                cut = cu > top_p
                cut[..., 1:] = cut[..., :-1].clone()
                cut[..., 0] = False
                sp[cut] = 0
                sp = sp / sp.sum(-1, keepdim=True)
                s = torch.multinomial(sp, 1).squeeze(-1)
                sampled = si.gather(-1, s.unsqueeze(-1)).squeeze(-1)
            else:
                sampled = torch.multinomial(probs, 1).squeeze(-1)
            ids[:, w] = digit_t[sampled]
        y_u = int(ID_TO_TOKEN[ids[0, 6].item()])
        d_t = int(ID_TO_TOKEN[ids[0, 7].item()])
        d_u = int(ID_TO_TOKEN[ids[0, 8].item()])
        return d_t * 10 + d_u, c.month_num, c.decade_int * 10 + y_u

    last = _one()
    for _ in range(max_tries):
        d, mo, y = _one()
        if valid_date(d, mo, y):
            return d, mo, y
        last = (d, mo, y)
    d, mo, y = last
    return min(max(d, 1), 28), mo, y


@torch.no_grad()
def gen_cgan(G, c, device):
    idx = cond_to_idx_t(c, device)
    z = torch.randn(1, G.noise_dim, device=device)
    soft = G(z, idx, tau=0.1, hard=True)
    s = soft.argmax(-1).item()
    return s // N_YEAR_UNITS + 1, c.month_num, c.decade_int * 10 + s % N_YEAR_UNITS


@torch.no_grad()
def gen_diffusion(model, c, device):
    T = model.T
    idx = cond_to_idx_t(c, device)
    d = torch.tensor([N_DAY], dtype=torch.long, device=device)
    y = torch.tensor([N_YEAR_UNITS], dtype=torch.long, device=device)
    for t in range(T, 0, -1):
        tt = torch.tensor([t], dtype=torch.long, device=device)
        dl, yl = model(d, y, idx, tt)
        ds = torch.distributions.Categorical(logits=dl).sample()
        ys = torch.distributions.Categorical(logits=yl).sample()
        up = 1.0 / t
        if d.item() == N_DAY and torch.rand(1, device=device).item() < up:
            d = ds
        if y.item() == N_YEAR_UNITS and torch.rand(1, device=device).item() < up:
            y = ys
    if d.item() == N_DAY or y.item() == N_YEAR_UNITS:
        tt = torch.ones(1, dtype=torch.long, device=device)
        dl, yl = model(d, y, idx, tt)
        if d.item() == N_DAY:
            d = torch.distributions.Categorical(logits=dl).sample()
        if y.item() == N_YEAR_UNITS:
            y = torch.distributions.Categorical(logits=yl).sample()
    return d.item() + 1, c.month_num, c.decade_int * 10 + y.item()


def load_model(name, device):
    here = Path(__file__).resolve().parent
    ckpt = torch.load(
        here / "weights" / f"{name}.pt", map_location=device, weights_only=False
    )
    cfg = ckpt["cfg"]
    if name == "cvae":
        m = CVAE(**cfg).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.eval()
    if name == "ar":
        m = ARTransformer(**cfg).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.eval()
    if name == "cgan":
        m = GeneratorNet(noise_dim=cfg["noise_dim"], hidden=cfg["hidden"]).to(device)
        m.load_state_dict(ckpt["state_dict_G"])
        return m.eval()
    if name == "diffusion":
        m = DiffusionNet(hidden=cfg["hidden"], T=cfg["T"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.eval()
    raise ValueError(name)


GENERATORS = {
    "cvae": lambda m, c, dev: gen_cvae(m, c, dev),
    "ar": lambda m, c, dev: gen_ar(m, c, dev),
    "cgan": lambda m, c, dev: gen_cgan(m, c, dev),
    "diffusion": lambda m, c, dev: gen_diffusion(m, c, dev),
}

DEFAULT_MODEL = "cvae"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(GENERATORS.keys()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    gen = GENERATORS[args.model]

    in_lines = [ln for ln in args.input.read_text().splitlines() if ln.strip()]
    out_lines = []
    for ln in in_lines:
        c = parse_condition_line(ln)
        d, mo, y = gen(model, c, device)
        out_lines.append(format_output_line(c, d, mo, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")
    print(
        f"[predict.py] model={args.model}  wrote {len(out_lines)} predictions -> {args.output}"
    )


if __name__ == "__main__":
    main()
