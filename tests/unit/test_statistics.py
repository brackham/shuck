import numpy as np

from shuck.statistics import robust_weighted_mean


def test_robust_weighted_mean_matches_inverse_variance_formula() -> None:
    data = np.array([[1.0, 5.0], [3.0, 9.0], [1000.0, 13.0]])
    variance = np.array([[1.0, 4.0], [3.0, 4.0], [1.0, 4.0]])

    result = robust_weighted_mean(data, variance, axis=0, sigma=8.0)

    np.testing.assert_allclose(result.mean, [1.5, 9.0])
    np.testing.assert_allclose(result.variance, [0.75, 4.0 / 3.0])
    np.testing.assert_array_equal(result.good[:, 0], [True, True, False])


def test_robust_weighted_mean_zero_mad_keeps_identical_values() -> None:
    data = np.array([1.0, 1.0, 1.0, 2.0])
    variance = np.ones(4)

    result = robust_weighted_mean(data, variance, sigma=8.0)

    assert result.mean == 1.0
    assert result.variance == 1.0 / 3.0
    np.testing.assert_array_equal(result.good, [True, True, True, False])


def test_robust_weighted_mean_excludes_invalid_variance_and_mask() -> None:
    data = np.array([1.0, 2.0, 3.0, np.nan])
    variance = np.array([1.0, 0.0, np.inf, 1.0])
    mask = np.array([True, True, True, True])

    result = robust_weighted_mean(data, variance, mask=mask)

    assert result.mean == 1.0
    assert result.variance == 1.0
    np.testing.assert_array_equal(result.good, [True, False, False, False])
