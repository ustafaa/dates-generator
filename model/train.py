#!/usr/bin/env python
"""Train one or all of the four research-backed models, persist weights + history."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.data import build_held_out_tuples, build_splits, load_records
from model.training import train_acgan, train_cvae, train_maskgit, train_mdlm


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_cvae(out_dir, model, hist, cfg):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "history": hist},
               out_dir / "cvae.pt")


def _save_acgan(out_dir, G, D, hist, cfg):
    torch.save({"state_dict_G": G.state_dict(), "state_dict_D": D.state_dict(),
                "cfg": cfg, "history": hist}, out_dir / "acgan.pt")


def _save_maskgit(out_dir, model, hist, cfg):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "history": hist},
               out_dir / "maskgit.pt")


def _save_mdlm(out_dir, model, hist, cfg):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "history": hist},
               out_dir / "mdlm.pt")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt", type=Path)
    p.add_argument("--out", default="model/weights", type=Path)
    p.add_argument("--models", nargs="+",
                   default=["cvae", "acgan", "maskgit", "mdlm"],
                   choices=["cvae", "acgan", "maskgit", "mdlm"])
    p.add_argument("--epochs-cvae", type=int, default=15)
    p.add_argument("--epochs-acgan", type=int, default=25)
    p.add_argument("--epochs-maskgit", type=int, default=15)
    p.add_argument("--epochs-mdlm", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-held-out", type=int, default=50)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    _set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[train.py] Loading {args.data}")
    records = load_records(args.data)
    if args.frac < 1.0:
        rng = random.Random(args.seed)
        records = rng.sample(records, int(len(records) * args.frac))
    print(f"[train.py] {len(records):,} records")

    in_set, held_out = build_held_out_tuples(records, n_held_out=args.n_held_out, seed=args.seed)
    train_recs, val_recs, test_recs = build_splits(in_set, val_frac=0.1, test_frac=0.1, seed=args.seed)
    print(f"[train.py] splits: train={len(train_recs):,}  val={len(val_recs):,}  "
          f"test={len(test_recs):,}  held_out_tuples={len(held_out):,}")

    summary = {}

    if "cvae" in args.models:
        cfg = {"latent_dim": 32, "hidden": 512, "cfg_dropout": 0.1}
        m, hist = train_cvae(train_recs, val_recs, epochs=args.epochs_cvae,
                             batch_size=args.batch_size, **cfg, seed=args.seed)
        _save_cvae(args.out, m, hist, cfg)
        summary["cvae"] = hist[-1]

    if "acgan" in args.models:
        runtime_cfg = {"noise_dim": 32, "hidden": 256, "tau_start": 1.0, "tau_end": 0.3,
                       "lambda_mle": 0.5, "lambda_gp": 10.0}
        G, D, hist = train_acgan(train_recs, val_recs, epochs=args.epochs_acgan,
                                  batch_size=args.batch_size, **runtime_cfg, seed=args.seed)
        save_cfg = {"noise_dim": runtime_cfg["noise_dim"], "hidden": runtime_cfg["hidden"]}
        _save_acgan(args.out, G, D, hist, save_cfg)
        summary["acgan"] = hist[-1]

    if "maskgit" in args.models:
        cfg = {"d_model": 128, "n_layers": 2, "n_heads": 4, "ff_dim": 256,
               "cfg_dropout": 0.1}
        m, hist = train_maskgit(train_recs, val_recs, epochs=args.epochs_maskgit,
                                batch_size=args.batch_size, **cfg, seed=args.seed)
        _save_maskgit(args.out, m, hist, cfg)
        summary["maskgit"] = hist[-1]

    if "mdlm" in args.models:
        cfg = {"hidden": 512, "T": 20, "cfg_dropout": 0.1}
        m, hist = train_mdlm(train_recs, val_recs, epochs=args.epochs_mdlm,
                             batch_size=args.batch_size, **cfg, seed=args.seed)
        _save_mdlm(args.out, m, hist, cfg)
        summary["mdlm"] = hist[-1]

    (args.out / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"[train.py] done. Summary -> {args.out / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
