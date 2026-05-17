from pathlib import Path

import pytest


@pytest.fixture
def tiny_data_text() -> str:
    return (
        "[WED] [JAN] [False] [180] 1-1-1800\n"
        "[THU] [JAN] [False] [180] 2-1-1800\n"
        "[FRI] [JAN] [False] [180] 3-1-1800\n"
        "[SAT] [JAN] [False] [180] 4-1-1800\n"
        "[SUN] [JAN] [True] [180] 1-1-1804\n"
        "[MON] [JAN] [True] [180] 2-1-1804\n"
        "[TUE] [JAN] [True] [180] 3-1-1804\n"
        "[WED] [JAN] [True] [180] 4-1-1804\n"
        "[THU] [JAN] [True] [180] 5-1-1804\n"
        "[FRI] [JAN] [True] [180] 6-1-1804\n"
    )


@pytest.fixture
def tiny_data_path(tmp_path, tiny_data_text) -> Path:
    p = tmp_path / "data.txt"
    p.write_text(tiny_data_text)
    return p
