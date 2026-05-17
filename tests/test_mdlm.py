import torch

from model.models.mdlm import MDLM, mdlm_loss
from model.tokenizer import N_DAY, N_JOINT, N_YEAR_UNITS


def test_forward_shape():
    torch.manual_seed(0)
    m = MDLM(hidden=32, T=5)
    x_in = torch.full((4,), m.MASK_ID, dtype=torch.long)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    t = torch.full((4,), 0.5)
    logits = m(x_in, cond_idx, t)
    assert logits.shape == (4, N_JOINT)


def test_sample_range():
    torch.manual_seed(0)
    m = MDLM(hidden=32, T=5)
    cond_idx = torch.zeros(32, 4, dtype=torch.long)
    d, yu = m.sample(cond_idx, w=1.0)
    assert d.shape == (32,) and yu.shape == (32,)
    assert int(d.min()) >= 0 and int(d.max()) < N_DAY
    assert int(yu.min()) >= 0 and int(yu.max()) < N_YEAR_UNITS


def test_loss_finite_and_backward():
    torch.manual_seed(0)
    m = MDLM(hidden=32, T=5)
    cond_idx = torch.zeros(8, 4, dtype=torch.long)
    joint = torch.zeros(8, dtype=torch.long)
    loss = mdlm_loss(m, cond_idx, joint)
    assert torch.isfinite(loss)
    loss.backward()
