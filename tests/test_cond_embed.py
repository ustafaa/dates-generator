import torch

from model.models.cond_embed import ConditionEmbedder


def test_shape():
    emb = ConditionEmbedder(embed_dim=32)
    assert emb.out_dim == 4 * 32
    cond_idx = torch.tensor([[0, 1, 0, 0], [3, 11, 1, 40]], dtype=torch.long)
    out = emb(cond_idx)
    assert out.shape == (2, 128)


def test_deterministic_under_seed():
    torch.manual_seed(0)
    emb1 = ConditionEmbedder()
    cond_idx = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    out1 = emb1(cond_idx).clone()
    torch.manual_seed(0)
    emb2 = ConditionEmbedder()
    out2 = emb2(cond_idx)
    assert torch.allclose(out1, out2)
