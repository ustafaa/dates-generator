from model.evaluation import baseline_random, baseline_smart_random, evaluate_records
from model.data import load_records


def test_baseline_random_runs(tiny_data_path):
    recs = load_records(tiny_data_path)
    rep = baseline_random(recs, seed=0)
    assert 0.0 <= rep.csr_all <= 1.0


def test_baseline_smart_random_perfect(tiny_data_path):
    recs = load_records(tiny_data_path)
    rep = baseline_smart_random(recs, seed=0)
    assert rep.csr_all == 1.0


def test_evaluate_records_perfect(tiny_data_path):
    recs = load_records(tiny_data_path)
    def perfect(cond_list):
        return [(d, m, y) for (c, d, m, y) in recs[:len(cond_list)]]
    rep = evaluate_records(perfect, recs)
    assert rep.csr_all == 1.0
