import numpy as np

from tools.run_rq4_informative_sensitivity import reciprocal_rank, same_value, value_slug


def test_reciprocal_rank_supports_positive_candidate_bags():
    scores = np.asarray([[0.9, 0.8, 0.1], [0.2, 0.9, 0.8]])
    labels = np.asarray([[0, 1, 1], [1, 0, 0]])
    # First positive ranks are 2 and 3: mean reciprocal rank = (1/2 + 1/3)/2.
    assert np.isclose(reciprocal_rank(scores, labels), (0.5 + 1 / 3) / 2 * 100)


def test_value_helpers_are_stable():
    assert same_value(0.2, 0.20000000000001)
    assert value_slug(0.001) == "0p001"
