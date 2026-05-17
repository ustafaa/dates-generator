import torch
import torch.nn.functional as F

from model.models.acgan import (
    Generator,
    Discriminator,
    gradient_penalty,
    acgan_g_step_loss,
    acgan_d_step_loss,
)
from model.tokenizer import N_JOINT


def test_generator_shape():
    torch.manual_seed(0)
    G = Generator(noise_dim=16, hidden=32)
    z = torch.randn(4, 16)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    x_soft = G(z, cond_idx, tau=1.0, hard=False)
    assert x_soft.shape == (4, N_JOINT)
    assert torch.allclose(x_soft.sum(-1), torch.ones(4), atol=1e-4)


def test_discriminator_shape():
    torch.manual_seed(0)
    D = Discriminator(hidden=32)
    x = torch.zeros(4, N_JOINT)
    x[:, 0] = 1.0
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    rf = D(x, cond_idx)
    assert rf.shape == (4,)


def test_gradient_penalty_finite():
    torch.manual_seed(0)
    D = Discriminator(hidden=32)
    x_real = F.one_hot(torch.zeros(4, dtype=torch.long), N_JOINT).float()
    x_fake = F.one_hot(torch.ones(4, dtype=torch.long), N_JOINT).float()
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    gp = gradient_penalty(D, x_real, x_fake, cond_idx)
    assert torch.isfinite(gp)


def test_g_step_loss_finite_and_backward():
    torch.manual_seed(0)
    G = Generator(noise_dim=16, hidden=32)
    D = Discriminator(hidden=32)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    joint = torch.zeros(4, dtype=torch.long)
    loss, _ = acgan_g_step_loss(G, D, cond_idx, joint, tau=1.0, lambda_mle=0.5)
    assert torch.isfinite(loss)
    loss.backward()


def test_d_step_loss_finite():
    torch.manual_seed(0)
    G = Generator(noise_dim=16, hidden=32)
    D = Discriminator(hidden=32)
    cond_idx = torch.zeros(4, 4, dtype=torch.long)
    joint = torch.zeros(4, dtype=torch.long)
    loss, _ = acgan_d_step_loss(G, D, cond_idx, joint, tau=1.0, lambda_gp=10.0)
    assert torch.isfinite(loss)
