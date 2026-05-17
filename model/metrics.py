"""Condition Satisfaction Rate (CSR), validity, and diversity metrics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from model.tokenizer import (
    Condition,
    NUM_TO_MON_NAME,
    date_weekday_token,
    is_leap_year,
    valid_date,
)


def check_dow(d: int, m: int, y: int, c: Condition) -> bool:
    return date_weekday_token(d, m, y) == c.dow


def check_mon(d: int, m: int, y: int, c: Condition) -> bool:
    return NUM_TO_MON_NAME[m] == c.mon


def check_leap(d: int, m: int, y: int, c: Condition) -> bool:
    return (c.leap == "[True]") == is_leap_year(y)


def check_dec(d: int, m: int, y: int, c: Condition) -> bool:
    return y // 10 == c.decade_int


@dataclass
class CSRReport:
    n: int
    n_valid: int
    valid_rate: float
    csr_all: float
    csr_dow: float
    csr_mon: float
    csr_leap: float
    csr_dec: float

    def __str__(self) -> str:
        return (
            f"n={self.n:<7d} valid={self.valid_rate:.3f}  "
            f"CSR_all={self.csr_all:.3f}  DOW={self.csr_dow:.3f}  "
            f"MON={self.csr_mon:.3f}  LEAP={self.csr_leap:.3f}  DEC={self.csr_dec:.3f}"
        )


def csr_report(generations: Iterable[Tuple[Condition, int, int, int]]) -> CSRReport:
    gens = list(generations)
    n = len(gens)
    if n == 0:
        return CSRReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n_valid = dow = mon = leap = dec = all_ok = 0
    for c, d, m, y in gens:
        if not valid_date(d, m, y):
            continue
        n_valid += 1
        a = check_dow(d, m, y, c)
        b = check_mon(d, m, y, c)
        e = check_leap(d, m, y, c)
        f = check_dec(d, m, y, c)
        dow += a
        mon += b
        leap += e
        dec += f
        all_ok += a and b and e and f
    return CSRReport(
        n=n,
        n_valid=n_valid,
        valid_rate=n_valid / n,
        csr_all=all_ok / n,
        csr_dow=dow / n,
        csr_mon=mon / n,
        csr_leap=leap / n,
        csr_dec=dec / n,
    )


def diversity_per_condition(
    generations: Iterable[Tuple[Condition, int, int, int]],
) -> Tuple[float, int]:
    by_c: dict = defaultdict(list)
    for c, d, m, y in generations:
        by_c[c].append((d, m, y))
    ratios: List[float] = []
    for samples in by_c.values():
        if len(samples) >= 2:
            ratios.append(len(set(samples)) / len(samples))
    if not ratios:
        return 0.0, 0
    return sum(ratios) / len(ratios), len(ratios)
