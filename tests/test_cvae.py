import torch

from model.models.cvae import CVAE, cvae_loss
from model.tokenizer import N_DAY, N_JOINT, N_YEAR_UNITS


def test_forward_shape():
    torch.manual_seed(0)
    m = CVAE(latent_dim=8, hidden=32)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    d = torch.zeros(4, dtype=torch.long)
    yu = torch.zeros(4, dtype=torch.long)
    logits, mu, logvar = m(cond_idx, d, yu)
    assert logits.shape == (4, N_JOINT)
    assert mu.shape == (4, 8) and logvar.shape == (4, 8)


def test_sample_range():
    torch.manual_seed(0)
    m = CVAE(latent_dim=8, hidden=32)
    cond_idx = torch.zeros(32, 4, dtype=torch.long)
    d, yu = m.sample(cond_idx, w=1.0)
    assert d.shape == (32,) and yu.shape == (32,)
    assert int(d.min()) >= 0 and int(d.max()) < N_DAY
    assert int(yu.min()) >= 0 and int(yu.max()) < N_YEAR_UNITS


def test_cfg_dropout_changes_output_distribution():
    torch.manual_seed(0)
    m = CVAE(latent_dim=8, hidden=32, cfg_dropout=1.0)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    d = torch.zeros(4, dtype=torch.long)
    yu = torch.zeros(4, dtype=torch.long)
    logits_a, _, _ = m(cond_idx, d, yu, training_cfg_dropout=True)
    logits_b, _, _ = m(cond_idx, d, yu, training_cfg_dropout=False)
    assert not torch.allclose(logits_a, logits_b)


def test_loss_finite_and_backward():
    torch.manual_seed(0)
    m = CVAE(latent_dim=8, hidden=32)
    cond_idx = torch.zeros(8, 4, dtype=torch.long)
    d = torch.zeros(8, dtype=torch.long)
    yu = torch.zeros(8, dtype=torch.long)
    logits, mu, logvar = m(cond_idx, d, yu)
    joint = d * N_YEAR_UNITS + yu
    loss, recon, kl = cvae_loss(logits, mu, logvar, joint, beta=1.0, free_bits=0.02)
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad
