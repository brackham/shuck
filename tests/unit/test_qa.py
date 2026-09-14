import numpy as np

from shuck.qa import _robust_y_limits


def test_robust_y_limits_pad_ordinary_asymmetric_distribution() -> None:
    values = np.linspace(-10.0, 30.0, 101)

    limits = _robust_y_limits(values)

    np.testing.assert_allclose(limits, (-13.52, 33.52))
    assert limits is not None
    assert not np.isclose(abs(limits[0]), abs(limits[1]))


def test_robust_y_limits_preserve_asymmetric_range() -> None:
    values = np.exp(np.linspace(0.0, 5.0, 1001))

    lower, upper = _robust_y_limits(values)

    assert lower < np.median(values) < upper
    assert not np.isclose(abs(lower), abs(upper))


def test_robust_y_limits_ignore_sparse_huge_outliers() -> None:
    values = np.concatenate((np.linspace(0.0, 10.0, 1001), [-1.0e9, 1.0e9]))

    lower, upper = _robust_y_limits(values)

    assert -1.0 < lower < 0.0
    assert 10.0 < upper < 11.0


def test_robust_y_limits_expand_constant_and_nearly_constant_data() -> None:
    np.testing.assert_allclose(_robust_y_limits(np.full(100, 3.0)), (2.7, 3.3))
    np.testing.assert_allclose(_robust_y_limits(np.zeros(100)), (-0.1, 0.1))
    lower, upper = _robust_y_limits(np.linspace(1.0, 1.0 + 1.0e-14, 100))
    np.testing.assert_allclose((lower, upper), (0.9, 1.1), rtol=1.0e-12)


def test_robust_y_limits_handle_nonfinite_and_small_samples() -> None:
    values = np.array([np.nan, -np.inf, 1.0, 2.0, 3.0, np.inf])

    np.testing.assert_allclose(_robust_y_limits(values), (0.8, 3.2))
    assert _robust_y_limits(np.array([np.nan, np.inf])) is None


def test_robust_y_limits_prefer_unmasked_values_and_fall_back_if_all_masked() -> None:
    values = np.concatenate((np.linspace(0.0, 10.0, 100), [1.0e9]))
    mask = np.zeros(values.shape, dtype=np.uint16)
    mask[-1] = 1

    lower, upper = _robust_y_limits(values, mask)

    np.testing.assert_allclose((lower, upper), (-0.88, 10.88))
    np.testing.assert_allclose(
        _robust_y_limits(np.array([1.0, 2.0, 3.0]), np.ones(3, dtype=np.uint16)),
        (0.8, 3.2),
    )
