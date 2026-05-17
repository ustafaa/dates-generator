"""Dataset, splits, and dataloader helpers."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from model.tokenizer import Condition, parse_data_line

Record = Tuple[Condition, int, int, int]


def load_records(path) -> List[Record]:
    out: List[Record] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        out.append(parse_data_line(line))
    return out


class DateDataset(Dataset):
    """Returns (cond_idx [4], day_idx, year_unit_idx) per item."""

    def __init__(self, records: List[Record]):
        cond_rows = [list(c.as_indices()) for c, *_ in records]
        day_rows = [d - 1 for _, d, _, _ in records]
        yu_rows = [y % 10 for _, _, _, y in records]
        self.cond_idx = torch.tensor(cond_rows, dtype=torch.long)
        self.day_idx = torch.tensor(day_rows, dtype=torch.long)
        self.yu_idx = torch.tensor(yu_rows, dtype=torch.long)

    def __len__(self) -> int:
        return self.cond_idx.size(0)

    def __getitem__(self, i: int):
        return self.cond_idx[i], self.day_idx[i], self.yu_idx[i]


def build_splits(
    records: List[Record],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Record], List[Record], List[Record]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = int(len(records) * val_frac)
    n_test = int(len(records) * test_frac)
    n_train = len(records) - n_val - n_test
    train = [records[i] for i in indices[:n_train]]
    val = [records[i] for i in indices[n_train : n_train + n_val]]
    test = [records[i] for i in indices[n_train + n_val :]]
    return train, val, test


def build_held_out_tuples(
    records: List[Record],
    n_held_out: int = 50,
    seed: int = 0,
) -> Tuple[List[Record], List[Record]]:
    """Reserve a held-out set whose condition tuples never appear in in_set."""
    rng = random.Random(seed)
    all_tuples = sorted({c.as_indices() for c, *_ in records})
    rng.shuffle(all_tuples)
    held_set = set(all_tuples[:n_held_out])
    in_set: List[Record] = []
    held: List[Record] = []
    for r in records:
        (held if r[0].as_indices() in held_set else in_set).append(r)
    return in_set, held
