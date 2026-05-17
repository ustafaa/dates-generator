from model.tokenizer import (
    Condition,
    parse_condition_line,
    parse_data_line,
    format_output_line,
    valid_date,
    is_leap_year,
    date_weekday_token,
    N_DOW, N_MON, N_LEAP, N_DEC, N_DAY, N_YEAR_UNITS, N_JOINT,
    DOW_TOKENS, MON_TOKENS, LEAP_TOKENS, DEC_TOKENS,
)


def test_dimensions():
    assert (N_DOW, N_MON, N_LEAP, N_DEC) == (7, 12, 2, 41)
    assert (N_DAY, N_YEAR_UNITS) == (31, 10)
    assert N_JOINT == 310
    assert DOW_TOKENS[0] == "[MON]"
    assert MON_TOKENS[-1] == "[DEC]"
    assert LEAP_TOKENS == ["[False]", "[True]"]
    assert DEC_TOKENS[0] == "[180]" and DEC_TOKENS[-1] == "[220]"


def test_parse_condition_line():
    c = parse_condition_line("[WED] [JAN] [False] [196] 3-12-1962")
    assert c == Condition("[WED]", "[JAN]", "[False]", "[196]")
    assert c.month_num == 1
    assert c.decade_int == 196


def test_parse_data_line():
    c, d, m, y = parse_data_line("[THU] [DEC] [True] [204] 3-12-2048")
    assert c.dow == "[THU]" and c.mon == "[DEC]"
    assert (d, m, y) == (3, 12, 2048)


def test_format_output_line():
    c = Condition("[WED]", "[JAN]", "[False]", "[180]")
    assert format_output_line(c, 1, 1, 1800) == "[WED] [JAN] [False] [180] 1-1-1800"


def test_valid_date():
    assert valid_date(28, 2, 2020) is True
    assert valid_date(29, 2, 2020) is True
    assert valid_date(29, 2, 2021) is False
    assert valid_date(31, 2, 2020) is False
    assert valid_date(0, 1, 2020) is False
    assert valid_date(31, 12, 1800) is True


def test_is_leap_year():
    assert is_leap_year(2000) is True
    assert is_leap_year(1900) is False
    assert is_leap_year(2024) is True
    assert is_leap_year(2023) is False


def test_dow_token():
    assert date_weekday_token(1, 1, 2024) == "[MON]"
    assert date_weekday_token(29, 2, 2020) == "[SAT]"
    assert date_weekday_token(1, 1, 1800) == "[WED]"
