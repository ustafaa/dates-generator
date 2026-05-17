#!/usr/bin/env python
"""Sweep classifier-free guidance weight w for the CVAE."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.data import build_held_out_tuples, build_splits, load_records
from model.evaluation import diversity_score, evaluate_records
from model.models.cvae import CVAE


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(REPO / "model" / "weights" / "cvae.pt",
                      map_location=device, weights_only=False)
    m = CVAE(**ckpt["cfg"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.train(False)

    records = load_records(REPO / "data" / "data.txt")
    in_set, _ = build_held_out_tuples(records, n_held_out=50, seed=0)
    _, val, _ = build_splits(in_set, val_frac=0.1, test_frac=0.1, seed=0)
    val_subset = val[:2000]

    weights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    rows = []
    for w in weights:
        def gen(conds, _w=w):
            rows_idx = torch.tensor([list(c.as_indices()) for c in conds], dtype=torch.long, device=device)
            d, yu = m.sample(rows_idx, w=_w)
            return [(int(d[i].item()) + 1, c.month_num, c.decade_int * 10 + int(yu[i].item()))
                    for i, c in enumerate(conds)]
        r = evaluate_records(gen, val_subset)
        div = diversity_score(gen, [c for c, *_ in val_subset[:200]], k=16)
        rows.append([w, r.csr_all, r.csr_dow, r.csr_leap, r.valid_rate, div])
        print(f"w={w:.1f}  CSR={r.csr_all:.3f}  DOW={r.csr_dow:.3f}  LEAP={r.csr_leap:.3f}  div={div:.3f}")

    out_csv = REPO / "results" / "cfg_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["w", "csr_all", "csr_dow", "csr_leap", "valid_rate", "diversity"])
        wr.writerows(rows)

    ws = [r[0] for r in rows]
    csrs = [r[1] for r in rows]
    divs = [r[5] for r in rows]
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))
    ax1.plot(ws, csrs, "o-", color="C0", label="CSR_all")
    ax1.set_xlabel("CFG weight w"); ax1.set_ylabel("CSR_all", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(ws, divs, "s--", color="C3", label="diversity")
    ax2.set_ylabel("diversity", color="C3")
    fig.tight_layout()
    fig.savefig(REPO / "results" / "figures" / "cfg_ablation.png", dpi=120)
    print(f"wrote {out_csv} and figure")


if __name__ == "__main__":
    main()
