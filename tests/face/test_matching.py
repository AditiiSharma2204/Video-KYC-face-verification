import numpy as np
import pytest

from videokyc.face.matching import cosine_distance, decide, similarity_score
from videokyc.face.models import Decision


def test_cosine_distance_identical_orthogonal_opposite():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_distance(a, a * 5) == pytest.approx(0.0, abs=1e-12)
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cosine_distance(a, -a) == pytest.approx(2.0)


def test_cosine_distance_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_distance(np.zeros(3), np.ones(3))


@pytest.mark.parametrize(
    "distance, expected",
    [
        (0.10, Decision.MATCH),
        (0.68 * 0.85, Decision.MATCH),  # inclusive lower edge of the band
        (0.68, Decision.REVIEW),  # exactly on the threshold is uncertain
        (0.68 * 1.15, Decision.REVIEW),  # inclusive upper edge
        (0.90, Decision.NO_MATCH),
    ],
)
def test_decide_bands(distance, expected):
    assert decide(distance, 0.68, review_margin=0.15) is expected


def test_zero_margin_is_binary():
    assert decide(0.67, 0.68, 0) is Decision.MATCH
    assert decide(0.69, 0.68, 0) is Decision.NO_MATCH


def test_invalid_margin():
    with pytest.raises(ValueError):
        decide(0.5, 0.68, 1.0)


def test_similarity_is_monotonic_and_bounded():
    scores = [similarity_score(d, 0.68) for d in (0.0, 0.34, 0.68, 1.36, 3.0)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100.0 and scores[2] == pytest.approx(50.0) and scores[-1] == 0.0
