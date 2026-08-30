import numpy as np
from types import SimpleNamespace

from tools.run_rq6_controlled_propagation import make_case, make_dataset, run_config, run_record_path


def test_controlled_case_has_consistent_truth():
    case = make_case(np.random.RandomState(7), node_num=8, window_size=10, max_lag=3)
    assert case.anomaly.shape == (10, 8)
    assert len(case.graphs) == 10
    assert int(case.relation.sum().item()) == int(case.response_mask.sum().item())
    assert int(case.response_mask[case.source].item()) == 0
    targets, sources = np.where(case.relation.numpy() > 0)
    assert all(1 <= int(case.lag[target, source]) <= 3 for target, source in zip(targets, sources))


def test_controlled_dataset_is_deterministic():
    left = make_dataset(11, 3)
    right = make_dataset(11, 3)
    assert [case.source for case in left] == [case.source for case in right]
    assert all(np.array_equal(a.anomaly.numpy(), b.anomaly.numpy()) for a, b in zip(left, right))


def test_formal_runs_have_resumable_per_seed_paths(tmp_path):
    args = SimpleNamespace(
        device="cpu", node_num=8, window_size=10, max_lag=3,
        train_cases=600, test_cases=240, epochs=35, batch_size=32,
        learning_rate=1e-3, noise=0.15,
    )
    config = run_config(args, "full", 2)
    assert config["variant"] == "full"
    assert config["seed"] == 2
    assert run_record_path(tmp_path, "full", 2) == tmp_path / "runs" / "full" / "seed2.json"
