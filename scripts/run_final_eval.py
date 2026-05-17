#!/usr/bin/env python
"""Final eval: CSR breakdown + diversity for all 4 models on three splits."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.data import build_held_out_tuples, build_splits, load_records
from model.evaluation import baseline_random, baseline_smart_random, diversity_score, evaluate_records
from model.models.acgan import Generator, acgan_sample
from model.models.cvae import CVAE
from model.models.maskgit import MaskGIT
from model.models.mdlm import MDLM


MODELS = ["cvae", "acgan", "maskgit", "mdlm"]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(name: str, weights_dir: Path, device):
    ckpt = torch.load(weights_dir / f"{name}.pt", map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    if name == "cvae":
        m = CVAE(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    if name == "acgan":
        m = Generator(**cfg).to(device); m.load_state_dict(ckpt["state_dict_G"])
        return m.train(False)
    if name == "maskgit":
        m = MaskGIT(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    if name == "mdlm":
        m = MDLM(**cfg).to(device); m.load_state_dict(ckpt["state_dict"])
        return m.train(False)
    raise ValueError(name)


def _make_gen(name, model, device, w=2.5):
    def gen_fn(conds):
        rows = [list(c.as_indices()) for c in conds]
        idx = torch.tensor(rows, dtype=torch.long, device=device)
        if name == "cvae":
            d, yu = model.sample(idx, w=w)
        elif name == "acgan":
            d, yu = acgan_sample(model, idx, tau=0.3)
        elif name == "maskgit":
            d, yu = model.sample(idx, w=w)
        elif name == "mdlm":
            d, yu = model.sample(idx, w=w)
        out = []
        for i, c in enumerate(conds):
            out.append((int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item())))
        return out
    return gen_fn


def main():
    device = _device()
    weights_dir = REPO / "model" / "weights"
    out_dir = REPO / "results"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(REPO / "data" / "data.txt")
    in_set, held = build_held_out_tuples(records, n_held_out=50, seed=0)
    train, val, test = build_splits(in_set, val_frac=0.1, test_frac=0.1, seed=0)
    splits = {"val": val, "test_random": test, "held_out_tuples": held}

    rows = []
    for split_name, recs in splits.items():
        r = baseline_random(recs[:5000], seed=0)
        rows.append(["random", split_name, r.csr_all, r.csr_dow, r.csr_mon, r.csr_leap, r.csr_dec, r.valid_rate])
        s = baseline_smart_random(recs[:5000], seed=0)
        rows.append(["smart_random", split_name, s.csr_all, s.csr_dow, s.csr_mon, s.csr_leap, s.csr_dec, s.valid_rate])

    div_summary = {}
    for name in MODELS:
        model = _load_model(name, weights_dir, device)
        gen = _make_gen(name, model, device)
        for split_name, recs in splits.items():
            r = evaluate_records(gen, recs[:5000])
            rows.append([name, split_name, r.csr_all, r.csr_dow, r.csr_mon, r.csr_leap, r.csr_dec, r.valid_rate])
        sample_conds = [c for c, *_ in val[:200]]
        div_summary[name] = diversity_score(gen, sample_conds, k=32)

    out_csv = out_dir / "csr_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "csr_all", "csr_dow", "csr_mon", "csr_leap", "csr_dec", "valid_rate"])
        w.writerows(rows)
    print(f"wrote {out_csv}")
    (out_dir / "diversity.json").write_text(json.dumps(div_summary, indent=2))

    # Training loss curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for name in MODELS:
        ck = torch.load(weights_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        hist = ck["history"]
        epochs = [h["epoch"] for h in hist]
        losses = [h.get("loss", h.get("g_loss", 0.0)) for h in hist]
        ax.plot(epochs, losses, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.legend()
    ax.set_title("Training loss per model")
    fig.tight_layout(); fig.savefig(figures_dir / "train_loss.png", dpi=120)

    # Val CSR curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for name in MODELS:
        ck = torch.load(weights_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        hist = ck["history"]
        epochs = [h["epoch"] for h in hist]
        csrs = [h["val_csr_all"] for h in hist]
        ax.plot(epochs, csrs, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("val CSR_all"); ax.legend()
    ax.set_title("Validation CSR_all per model")
    fig.tight_layout(); fig.savefig(figures_dir / "val_csr.png", dpi=120)

    # Final bar chart
    val_rows = [r for r in rows if r[1] == "val" and r[0] in MODELS]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar([r[0] for r in val_rows], [r[2] for r in val_rows])
    ax.set_ylabel("val CSR_all"); ax.set_ylim(0, 1.0)
    ax.set_title("Final val CSR_all per model")
    fig.tight_layout(); fig.savefig(figures_dir / "final_csr.png", dpi=120)
    print(f"wrote figures -> {figures_dir}")


if __name__ == "__main__":
    main()
