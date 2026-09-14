"""Wavelength-dependent point-source tracing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from shuck.calibration.flat import _robust_polynomial_fit
from shuck.calibration.rectify import RectifiedOrder
from shuck.extraction.profile import ApertureLocation, fit_spatial_peak


@dataclass(frozen=True)
class SpectralTrace:
    """Polynomial source position versus wavelength for one order."""

    order: int
    coefficients: np.ndarray
    sample_wavelength_micron: np.ndarray
    sample_position_arcsec: np.ndarray
    sample_used: np.ndarray
    rms_arcsec: float
    valid: bool

    def position(self, wavelength_micron: np.ndarray) -> np.ndarray:
        """Evaluate the trace position in arcseconds."""

        return np.polynomial.polynomial.polyval(wavelength_micron, self.coefficients)


def trace_order(
    order: RectifiedOrder,
    aperture: ApertureLocation,
    *,
    degree: int = 2,
    step: int = 7,
    sum_width: int = 7,
    window_threshold_arcsec: float = 5.0,
) -> SpectralTrace:
    """Trace a point source following SpeXTool ``mc_tracespec.pro``."""

    if order.order != aperture.order:
        raise ValueError("rectified order and aperture order do not match")
    half_width = sum_width // 2
    columns = np.arange(step - 1, order.image.shape[1] - step + 1, step, dtype=int)
    wavelengths = order.wavelength_micron[columns]
    positions = np.full(len(columns), np.nan)
    for index, column in enumerate(columns):
        low = max(0, column - half_width)
        high = min(order.image.shape[1], column + half_width + 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            collapsed = np.nanmedian(
                np.where(order.mask[:, low:high] == 0, order.image[:, low:high], np.nan),
                axis=1,
            )
        center, _, _, found = fit_spatial_peak(
            order.spatial_arcsec,
            aperture.sign * collapsed,
            guess=aperture.position_arcsec,
            window_radius=1.0,
            required_sign=1,
        )
        if found and abs(center - aperture.position_arcsec) <= window_threshold_arcsec:
            positions[index] = center
    finite = np.isfinite(wavelengths) & np.isfinite(positions)
    if np.count_nonzero(finite) < degree + 1:
        return SpectralTrace(
            order=order.order,
            coefficients=np.full(degree + 1, np.nan),
            sample_wavelength_micron=wavelengths,
            sample_position_arcsec=positions,
            sample_used=np.zeros(len(columns), dtype=bool),
            rms_arcsec=np.nan,
            valid=False,
        )
    coefficients = _robust_polynomial_fit(
        wavelengths[finite],
        positions[finite],
        degree,
        threshold=3.0,
        fractional_sigma_change=0.01,
    )
    residual = positions - np.polynomial.polynomial.polyval(wavelengths, coefficients)
    center = np.nanmedian(residual[finite])
    sigma = 1.482 * np.nanmedian(np.abs(residual[finite] - center))
    used = finite & ((np.abs(residual - center) <= 3 * sigma) if sigma > 0 else residual == center)
    if np.count_nonzero(used) < degree + 1:
        used = finite
    coefficients = np.polynomial.polynomial.polyfit(wavelengths[used], positions[used], degree)
    residual = positions - np.polynomial.polynomial.polyval(wavelengths, coefficients)
    rms = float(np.sqrt(np.mean(residual[used] ** 2)))
    return SpectralTrace(
        order=order.order,
        coefficients=coefficients,
        sample_wavelength_micron=wavelengths,
        sample_position_arcsec=positions,
        sample_used=used,
        rms_arcsec=rms,
        valid=True,
    )


def trace_orders(
    orders: tuple[RectifiedOrder, ...],
    apertures: tuple[ApertureLocation, ...],
    *,
    degree: int = 2,
) -> tuple[SpectralTrace, ...]:
    """Trace one aperture through every order via ``mc_tracespec.pro``."""

    aperture_by_order = {aperture.order: aperture for aperture in apertures}
    return tuple(
        trace_order(order, aperture_by_order[order.order], degree=degree) for order in orders
    )
