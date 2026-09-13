import numpy as np

from shuck.calibration.wavecal import (
    _robust_fit_wavelength,
    cross_correlation_offset,
    evaluate_polynomial_2d,
)


def test_evaluate_polynomial_2d_uses_spextool_coefficient_order() -> None:
    # c00, c10, c20, c01, c11, c21
    coefficients = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 11.0])
    x = np.array([2.0, 3.0])
    y = np.array([10.0, 20.0])
    expected = 1 + 2 * x + 3 * x**2 + 5 * y + 7 * x * y + 11 * x**2 * y

    np.testing.assert_allclose(evaluate_polynomial_2d(x, y, 2, 1, coefficients), expected)


def test_cross_correlation_offset_recovers_subpixel_shift() -> None:
    x = np.arange(200, dtype=float)
    anchor = np.exp(-0.5 * ((x - 70) / 2) ** 2) + 0.5 * np.exp(-0.5 * ((x - 130) / 4) ** 2)
    observed = np.interp(x - 2.4, x, anchor, left=0.0, right=0.0)

    offset = cross_correlation_offset(x, anchor, x, observed)

    np.testing.assert_allclose(offset, 2.4, atol=0.25)


def test_robust_wavelength_fit_is_stable_in_raw_detector_coordinates() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(400, 1600, 100)
    order = rng.integers(387, 418, 100).astype(float)
    coefficients = np.array(
        [1.2, 5e-5, -2e-8, 4e-12, 2e-4, -1e-7, 3e-11, 0.0, -3e-7, 2e-10, 0.0, 0.0]
    )
    wavelength = evaluate_polynomial_2d(x, order, 3, 2, coefficients)
    wavelength += rng.normal(0, 1e-9, 100)
    wavelength[0] += 1e-3

    fitted, good = _robust_fit_wavelength(x, order, wavelength, 3, 2)

    assert not good[0]
    np.testing.assert_allclose(
        evaluate_polynomial_2d(x[good], order[good], 3, 2, fitted),
        wavelength[good],
        atol=5e-9,
    )
