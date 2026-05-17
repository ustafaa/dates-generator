#!/usr/bin/env python
"""CLI entry point -- matches the assignment spec:

    python predict.py -i $path_to_input -o $path_to_output [--model cvae|acgan|maskgit|mdlm]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.models.acgan import Generator, acgan_sample
from model.models.cvae import CVAE
from model.models.maskgit import MaskGIT
from model.models.mdlm import MDLM
from model.tokenizer import (
    Condition,
    format_output_line,
    parse_condition_line,
    valid_date,
)

DEFAULT_MODEL = "cvae"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(name: str, weights_dir: Path, device):
    ckpt_path = weights_dir / f"{name}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    if name == "cvae":
        m = CVAE(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    if name == "acgan":
        m = Generator(**{k: v for k, v in cfg.items() if k in {"noise_dim", "hidden", "embed_dim"}}).to(device)
        m.load_state_dict(ckpt["state_dict_G"]); return m.train(False)
    if name == "maskgit":
        m = MaskGIT(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    if name == "mdlm":
        m = MDLM(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    raise ValueError(f"Unknown model: {name}")


def _cond_to_idx(c, device):
    return torch.tensor([list(c.as_indices())], dtype=torch.long, device=device)


@torch.no_grad()
def _gen_cvae(m, c, device, w: float = 2.5):
    idx = _cond_to_idx(c, device)
    d, yu = m.sample(idx, w=w)
    return int(d.item()) + 1, c.month_num, c.decade_int * 10 + int(yu.item())


@torch.no_grad()
def _gen_acgan(m, c, device, tau: float = 0.3):
    idx = _cond_to_idx(c, device)
    d, yu = acgan_sample(m, idx, tau=tau)
    return int(d.item()) + 1, c.month_num, c.decade_int * 10 + int(yu.item())


@torch.no_grad()
def _gen_maskgit(m, c, device, w: float = 2.5):
    idx = _cond_to_idx(c, device)
    d, yu = m.sample(idx, w=w)
    return int(d.item()) + 1, c.month_num, c.decade_int * 10 + int(yu.item())


@torch.no_grad()
def _gen_mdlm(m, c, device, w: float = 2.5):
    idx = _cond_to_idx(c, device)
    d, yu = m.sample(idx, w=w)
    return int(d.item()) + 1, c.month_num, c.decade_int * 10 + int(yu.item())


GENERATORS: dict[str, Callable] = {
    "cvae": _gen_cvae,
    "acgan": _gen_acgan,
    "maskgit": _gen_maskgit,
    "mdlm": _gen_mdlm,
}


def _retry_for_validity(gen_fn, model, c, device, max_tries: int = 5):
    last = None
    for _ in range(max_tries):
        d, mo, y = gen_fn(model, c, device)
        last = (d, mo, y)
        if valid_date(d, mo, y):
            return d, mo, y
    d, mo, y = last
    return min(max(d, 1), 28), mo, y


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(GENERATORS.keys()))
    p.add_argument("--weights-dir",
                   default=str(Path(__file__).resolve().parent / "weights"),
                   type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    _set_seed(args.seed)
    device = _device()
    model = _load(args.model, args.weights_dir, device)
    gen = GENERATORS[args.model]

    in_lines = [ln for ln in args.input.read_text().splitlines() if ln.strip()]
    out_lines = []
    for ln in in_lines:
        c = parse_condition_line(ln)
        d, mo, y = _retry_for_validity(gen, model, c, device, max_tries=5)
        out_lines.append(format_output_line(c, d, mo, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n")
    print(f"[predict.py] model={args.model}  wrote {len(out_lines)} predictions -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
