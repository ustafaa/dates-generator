from model.data import load_records
from model.training import train_cvae, train_acgan, train_maskgit, train_mdlm


def _recs(tiny_data_path):
    return load_records(tiny_data_path)


def test_train_cvae_smoke(tiny_data_path):
    recs = _recs(tiny_data_path)
    m, hist = train_cvae(recs, recs, epochs=1, batch_size=4, hidden=16,
                        latent_dim=4, val_eval_n=10, verbose=False)
    assert len(hist) == 1


def test_train_acgan_smoke(tiny_data_path):
    recs = _recs(tiny_data_path)
    G, D, hist = train_acgan(recs, recs, epochs=1, batch_size=4,
                              noise_dim=8, hidden=16, val_eval_n=10, verbose=False)
    assert len(hist) == 1


def test_train_maskgit_smoke(tiny_data_path):
    recs = _recs(tiny_data_path)
    m, hist = train_maskgit(recs, recs, epochs=1, batch_size=4, d_model=16,
                            n_layers=1, n_heads=2, ff_dim=32, val_eval_n=10, verbose=False)
    assert len(hist) == 1


def test_train_mdlm_smoke(tiny_data_path):
    recs = _recs(tiny_data_path)
    m, hist = train_mdlm(recs, recs, epochs=1, batch_size=4, hidden=16, T=4,
                        val_eval_n=10, verbose=False)
    assert len(hist) == 1
