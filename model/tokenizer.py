"""Tokenizer, parsers, vocab, calendar utilities for the dates-generator task."""
from __future__ import annotations

import calendar
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Tuple

DOW_TOKENS = ["[MON]", "[TUE]", "[WED]", "[THU]", "[FRI]", "[SAT]", "[SUN]"]
MON_TOKENS = [
    "[JAN]", "[FEB]", "[MAR]", "[APR]", "[MAY]", "[JUN]",
    "[JUL]", "[AUG]", "[SEP]", "[OCT]", "[NOV]", "[DEC]",
]
LEAP_TOKENS = ["[False]", "[True]"]
DEC_TOKENS = [f"[{d}]" for d in range(180, 221)]

DOW_TO_IDX = {t: i for i, t in enumerate(DOW_TOKENS)}
MON_TO_IDX = {t: i for i, t in enumerate(MON_TOKENS)}
LEAP_TO_IDX = {t: i for i, t in enumerate(LEAP_TOKENS)}
DEC_TO_IDX = {t: i for i, t in enumerate(DEC_TOKENS)}
MON_NAME_TO_NUM = {t: i + 1 for i, t in enumerate(MON_TOKENS)}
NUM_TO_MON_NAME = {v: k for k, v in MON_NAME_TO_NUM.items()}

N_DOW, N_MON, N_LEAP, N_DEC = 7, 12, 2, 41
N_DAY = 31
N_YEAR_UNITS = 10
N_JOINT = N_DAY * N_YEAR_UNITS

CONDITION_RE = re.compile(
    r"^\[(?P<dow>[A-Z]{3})\]\s+\[(?P<mon>[A-Z]{3})\]\s+"
    r"\[(?P<leap>True|False)\]\s+\[(?P<dec>\d{3})\]"
)


@dataclass(frozen=True)
class Condition:
    dow: str
    mon: str
    leap: str
    dec: str

    def as_indices(self) -> Tuple[int, int, int, int]:
        return (
            DOW_TO_IDX[self.dow],
            MON_TO_IDX[self.mon],
            LEAP_TO_IDX[self.leap],
            DEC_TO_IDX[self.dec],
        )

    @property
    def month_num(self) -> int:
        return MON_NAME_TO_NUM[self.mon]

    @property
    def decade_int(self) -> int:
        return int(self.dec.strip("[]"))

    def as_prefix(self) -> str:
        return f"{self.dow} {self.mon} {self.leap} {self.dec}"


def parse_condition_line(line: str) -> Condition:
    m = CONDITION_RE.match(line.strip())
    if not m:
        raise ValueError(f"Malformed condition line: {line!r}")
    return Condition(
        f"[{m['dow']}]", f"[{m['mon']}]", f"[{m['leap']}]", f"[{m['dec']}]"
    )


def parse_data_line(line: str) -> Tuple[Condition, int, int, int]:
    cond = parse_condition_line(line)
    date_str = line.strip().rsplit(maxsplit=1)[-1]
    d_s, m_s, y_s = date_str.split("-")
    return cond, int(d_s), int(m_s), int(y_s)


def format_output_line(cond: Condition, day: int, month: int, year: int) -> str:
    return f"{cond.as_prefix()} {day}-{month}-{year}"


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def valid_date(day: int, month: int, year: int) -> bool:
    if not (1 <= month <= 12) or year < 1:
        return False
    return 1 <= day <= calendar.monthrange(year, month)[1]


def date_weekday_token(day: int, month: int, year: int) -> str:
    return DOW_TOKENS[_dt.date(year, month, day).weekday()]
