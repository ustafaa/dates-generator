import torch

from model.models.maskgit import MaskGIT, maskgit_loss
from model.tokenizer import N_DAY, N_YEAR_UNITS


def test_forward_shape():
    torch.manual_seed(0)
    m = MaskGIT(d_model=32, n_layers=1, n_heads=2, ff_dim=64)
    d_in = torch.full((4,), m.DAY_MASK, dtype=torch.long)
    yu_in = torch.full((4,), m.YU_MASK, dtype=torch.long)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    d_l, yu_l = m(d_in, yu_in, cond_idx)
    assert d_l.shape == (4, N_DAY)
    assert yu_l.shape == (4, N_YEAR_UNITS)


def test_sample_range():
    torch.manual_seed(0)
    m = MaskGIT(d_model=32, n_layers=1, n_heads=2, ff_dim=64)
    cond_idx = torch.zeros(32, 4, dtype=torch.long)
    d, yu = m.sample(cond_idx, w=1.0)
    assert d.shape == (32,) and yu.shape == (32,)
    assert int(d.min()) >= 0 and int(d.max()) < N_DAY
    assert int(yu.min()) >= 0 and int(yu.max()) < N_YEAR_UNITS


def test_loss_finite_and_backward():
    torch.manual_seed(0)
    m = MaskGIT(d_model=32, n_layers=1, n_heads=2, ff_dim=64)
    cond_idx = torch.zeros(8, 4, dtype=torch.long)
    d = torch.zeros(8, dtype=torch.long)
    yu = torch.zeros(8, dtype=torch.long)
    loss = maskgit_loss(m, cond_idx, d, yu)
    assert torch.isfinite(loss)
    loss.backward()
