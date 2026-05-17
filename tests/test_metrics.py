from model.tokenizer import Condition
from model.metrics import (
    csr_report,
    diversity_per_condition,
    check_dow, check_mon, check_leap, check_dec,
)


def _cond(dow="[MON]", mon="[JAN]", leap="[False]", dec="[200]"):
    return Condition(dow, mon, leap, dec)


def test_check_atomic():
    c = _cond()
    assert check_dow(7, 1, 2002, c)
    assert check_mon(7, 1, 2002, c)
    assert check_leap(7, 1, 2002, c)
    assert check_dec(7, 1, 2002, c)


def test_csr_perfect():
    c = _cond()
    gens = [(c, 7, 1, 2002)] * 5
    r = csr_report(gens)
    assert r.csr_all == 1.0
    assert r.valid_rate == 1.0


def test_csr_zero_invalid():
    c = _cond()
    gens = [(c, 31, 2, 2002)]
    r = csr_report(gens)
    assert r.valid_rate == 0.0
    assert r.csr_all == 0.0


def test_csr_partial():
    c = _cond()
    gens = [(c, 8, 1, 2002), (c, 7, 1, 2002)]
    r = csr_report(gens)
    assert r.csr_dow == 0.5
    assert r.csr_mon == 1.0
    assert r.csr_all == 0.5


def test_diversity():
    c1 = _cond(dec="[200]")
    c2 = _cond(dec="[201]")
    gens = [
        (c1, 1, 1, 2000), (c1, 1, 1, 2000),
        (c2, 1, 1, 2010), (c2, 2, 1, 2010),
    ]
    mean_div, n = diversity_per_condition(gens)
    assert n == 2
    assert mean_div == 0.75
