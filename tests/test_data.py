import torch
from torch.utils.data import DataLoader

from model.data import DateDataset, load_records, build_splits, build_held_out_tuples


def test_load_records(tiny_data_path):
    recs = load_records(tiny_data_path)
    assert len(recs) == 10
    assert recs[0][0].dow == "[WED]"
    assert recs[0][1:] == (1, 1, 1800)


def test_dataset(tiny_data_path):
    recs = load_records(tiny_data_path)
    ds = DateDataset(recs)
    assert len(ds) == 10
    cond, d, yu = ds[0]
    assert cond.shape == (4,) and cond.dtype == torch.long
    assert int(d) == 0
    assert int(yu) == 0


def test_dataloader(tiny_data_path):
    recs = load_records(tiny_data_path)
    ds = DateDataset(recs)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    cond, d, yu = next(iter(loader))
    assert cond.shape == (4, 4)
    assert d.shape == (4,) and yu.shape == (4,)


def test_build_splits(tiny_data_path):
    recs = load_records(tiny_data_path)
    train, val, test = build_splits(recs, val_frac=0.2, test_frac=0.2, seed=0)
    assert len(train) + len(val) + len(test) == 10
    assert len(val) == 2 and len(test) == 2
    train2, val2, test2 = build_splits(recs, val_frac=0.2, test_frac=0.2, seed=0)
    assert train == train2 and val == val2 and test == test2


def test_held_out_tuples(tiny_data_path):
    recs = load_records(tiny_data_path)
    in_set, held = build_held_out_tuples(recs, n_held_out=2, seed=0)
    assert len(in_set) + len(held) == 10
    held_keys = {tuple(c.as_indices()) for c, *_ in held}
    in_keys = {tuple(c.as_indices()) for c, *_ in in_set}
    assert held_keys.isdisjoint(in_keys)
