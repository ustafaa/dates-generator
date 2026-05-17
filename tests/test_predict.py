import subprocess
import sys
from pathlib import Path

import pytest
import torch

from model.models.cvae import CVAE


@pytest.fixture
def stub_cvae_weights(tmp_path) -> Path:
    torch.manual_seed(0)
    m = CVAE(latent_dim=8, hidden=32, cfg_dropout=0.1)
    ckpt = {
        "state_dict": m.state_dict(),
        "cfg": {"latent_dim": 8, "hidden": 32, "cfg_dropout": 0.1},
    }
    weights = tmp_path / "cvae.pt"
    torch.save(ckpt, weights)
    return weights


def test_predict_runs_and_produces_correct_line_count(tmp_path, stub_cvae_weights):
    in_path = tmp_path / "in.txt"
    in_path.write_text("[WED] [JAN] [False] [180]\n[MON] [JAN] [False] [190]\n")
    out_path = tmp_path / "out.txt"
    weights_dir = stub_cvae_weights.parent
    repo = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, str(repo / "model" / "predict.py"),
         "-i", str(in_path), "-o", str(out_path),
         "--model", "cvae", "--weights-dir", str(weights_dir),
         "--seed", "0"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    out_lines = [l for l in out_path.read_text().splitlines() if l.strip()]
    assert len(out_lines) == 2
    for line in out_lines:
        assert line.count("[") == 4 and line.count("]") == 4
        assert line.split()[-1].count("-") == 2
