"""Comprehensive evaluation: baselines, CSR breakdown, diversity."""
from __future__ import annotations

import calendar
import random
from typing import Callable, List

from model.data import Record
from model.metrics import CSRReport, check_dow, check_leap, csr_report, diversity_per_condition
from model.tokenizer import Condition


def _random_date(cond: Condition, rng: random.Random):
    month = cond.month_num
    yu = rng.randint(0, 9)
    year = cond.decade_int * 10 + yu
    days = calendar.monthrange(year, month)[1]
    return rng.randint(1, days), month, year


def _smart_random(cond: Condition, rng: random.Random, max_tries: int = 200):
    last = _random_date(cond, rng)
    for _ in range(max_tries):
        d, m, y = _random_date(cond, rng)
        if check_dow(d, m, y, cond) and check_leap(d, m, y, cond):
            return d, m, y
        last = (d, m, y)
    return last


def baseline_random(records: List[Record], seed: int = 0) -> CSRReport:
    rng = random.Random(seed)
    gens = [(c, *_random_date(c, rng)) for c, *_ in records]
    return csr_report(gens)


def baseline_smart_random(records: List[Record], seed: int = 0) -> CSRReport:
    rng = random.Random(seed)
    gens = [(c, *_smart_random(c, rng)) for c, *_ in records]
    return csr_report(gens)


def evaluate_records(
    gen_fn: Callable[[List[Condition]], List[tuple]],
    records: List[Record],
) -> CSRReport:
    conds = [c for c, *_ in records]
    out = gen_fn(conds)
    gens = [(c, d, m, y) for c, (d, m, y) in zip(conds, out)]
    return csr_report(gens)


def diversity_score(
    gen_fn: Callable[[List[Condition]], List[tuple]],
    conds: List[Condition],
    k: int = 32,
) -> float:
    samples = []
    for _ in range(k):
        out = gen_fn(conds)
        for c, (d, m, y) in zip(conds, out):
            samples.append((c, d, m, y))
    mean_div, _ = diversity_per_condition(samples)
    return mean_div
