"""Spatial-profile and automatic aperture-location calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from shuck.calibration.rectify import RectifiedOrder
from shuck.statistics import robust_weighted_mean


@dataclass(frozen=True)
class SpatialProfile:
    """Median-background-subtracted average slit profile for one order."""

    order: int
    spatial_arcsec: np.ndarray
    normalized_flux: np.ndarray


@dataclass(frozen=True)
class ApertureLocation:
    """Automatically fitted point-source location for one order."""

    order: int
    position_arcsec: float
    sign: int
    fwhm_arcsec: float
    valid: bool


def make_spatial_profile(order: RectifiedOrder) -> SpatialProfile:
    """Construct an average spatial profile as ``mc_mkspatprof.pro`` does."""

    image = np.asarray(order.image, dtype=np.float64)
    good = (order.mask == 0) & np.isfinite(image)
    background = np.nanmedian(np.where(good, image, np.nan), axis=0)
    residual = image - background[None, :]
    combined = robust_weighted_mean(
        residual.T,
        np.ones_like(residual.T),
        axis=0,
        sigma=4.0,
        mask=good.T,
    ).mean
    normalization = np.nansum(np.abs(combined))
    profile = combined / normalization if normalization > 0 else np.full_like(combined, np.nan)
    return SpatialProfile(
        order=order.order,
        spatial_arcsec=order.spatial_arcsec.copy(),
        normalized_flux=profile,
    )


def make_spatial_profiles(orders: tuple[RectifiedOrder, ...]) -> tuple[SpatialProfile, ...]:
    """Construct all order profiles following SpeXTool ``mc_mkspatprof.pro``."""

    return tuple(make_spatial_profile(order) for order in orders)


def _gaussian_with_constant(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    constant: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + constant


def fit_spatial_peak(
    spatial: np.ndarray,
    flux: np.ndarray,
    *,
    guess: float | None = None,
    window_radius: float | None = None,
    required_sign: int | None = None,
) -> tuple[float, float, float, bool]:
    """Fit the four-term Gaussian used by SpeXTool ``mc_findpeaks.pro``."""

    x = np.asarray(spatial, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if guess is not None and window_radius is not None:
        finite &= np.abs(x - guess) <= window_radius
    if np.count_nonzero(finite) < 5:
        return np.nan, np.nan, np.nan, False
    x = x[finite]
    y = y[finite]
    baseline = float(np.median(y))
    if required_sign not in {None, -1, 1}:
        raise ValueError("required_sign must be -1, 1, or None")
    signal = np.abs(y - baseline) if required_sign is None else required_sign * (y - baseline)
    guess_position = float(x[np.argmax(signal)]) if guess is None else float(guess)
    amplitude = float(np.interp(guess_position, x, signal))
    spacing = float(np.median(np.diff(x)))
    try:
        parameters, _ = optimize.curve_fit(
            _gaussian_with_constant,
            x,
            signal,
            p0=(amplitude, guess_position, 0.5, 0.0),
            bounds=(
                [0.0, x.min(), spacing / 4, -np.inf],
                [np.inf, x.max(), max(np.ptp(x), spacing), np.inf],
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, False
    center = float(parameters[1])
    fwhm = float(2.354820045 * abs(parameters[2]))
    signed_value = float(np.interp(center, x, y))
    sign = (
        (1.0 if signed_value >= baseline else -1.0)
        if required_sign is None
        else float(required_sign)
    )
    valid = x.min() <= center <= x.max() and np.isfinite(fwhm) and fwhm > 0
    return center, sign, fwhm, valid


def find_apertures(profiles: tuple[SpatialProfile, ...]) -> tuple[ApertureLocation, ...]:
    """Automatically find one aperture per order via ``mc_findpeaks.pro``."""

    if not profiles:
        return ()
    reference_grid = profiles[0].spatial_arcsec
    if any(not np.array_equal(profile.spatial_arcsec, reference_grid) for profile in profiles[1:]):
        raise ValueError("spatial-profile grids differ between orders")
    aggregate = np.nanmedian(
        np.stack([profile.normalized_flux for profile in profiles]),
        axis=0,
    )
    global_position, global_sign, global_fwhm, global_valid = fit_spatial_peak(
        reference_grid,
        aggregate,
        required_sign=1,
    )
    global_valid = global_valid and global_sign == 1
    locations: list[ApertureLocation] = []
    for profile in profiles:
        position, sign, fwhm, valid = fit_spatial_peak(
            profile.spatial_arcsec,
            profile.normalized_flux,
            guess=global_position,
            window_radius=1.0,
            required_sign=1,
        )
        # The accepted path contains one positive aperture.  In weak orders,
        # use SpeXTool's fixed-position behavior with the automatically
        # measured cross-order position instead of selecting an unrelated
        # negative or edge feature.
        if not valid or sign != 1:
            position = global_position
            sign = global_sign
            fwhm = global_fwhm
            valid = global_valid
        locations.append(
            ApertureLocation(
                order=profile.order,
                position_arcsec=position,
                sign=int(sign) if np.isfinite(sign) else 0,
                fwhm_arcsec=fwhm,
                valid=valid,
            )
        )
    return tuple(locations)
