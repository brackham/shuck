"""Order merging for telluric-corrected spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from shuck.extraction.optimal import ExtractedOrder


@dataclass(frozen=True)
class MergedSpectrum:
    """One continuous spectrum assembled from adjacent echelle orders."""

    wavelength_micron: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray
    orders: tuple[int, ...]
    overlap_ranges: tuple[tuple[float, float] | None, ...]


def _trim_order(order: ExtractedOrder) -> tuple[np.ndarray, ...]:
    finite = np.isfinite(order.wavelength_micron) & np.isfinite(order.flux)
    if not np.any(finite):
        raise ValueError(f"order {order.order} has no finite spectrum")
    indices = np.flatnonzero(finite)
    selection = slice(indices[0], indices[-1] + 1)
    arrays = (
        np.asarray(order.wavelength_micron[selection], dtype=np.float64),
        np.asarray(order.flux[selection], dtype=np.float64),
        np.asarray(order.uncertainty[selection], dtype=np.float64),
        np.asarray(order.mask[selection], dtype=np.uint16),
    )
    direction = np.nanmedian(np.diff(arrays[0]))
    if direction == 0 or not np.isfinite(direction):
        raise ValueError(f"order {order.order} has a non-monotonic wavelength grid")
    if direction < 0:
        arrays = tuple(array[::-1] for array in arrays)
    if np.any(np.diff(arrays[0]) <= 0):
        raise ValueError(f"order {order.order} has a non-monotonic wavelength grid")
    return arrays


def _interpolate_order(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    output_wavelength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample values, errors, and flags as ``mc_interpspec.pro``."""

    output_flux = np.full(output_wavelength.shape, np.nan)
    output_uncertainty = np.full(output_wavelength.shape, np.nan)
    output_mask = np.zeros(output_wavelength.shape, dtype=np.uint16)
    inside = (output_wavelength >= wavelength[0]) & (output_wavelength <= wavelength[-1])
    positions = np.flatnonzero(inside)
    if positions.size == 0:
        return output_flux, output_uncertainty, output_mask
    right = np.searchsorted(wavelength, output_wavelength[inside], side="left")
    right = np.clip(right, 0, len(wavelength) - 1)
    exact = wavelength[right] == output_wavelength[inside]
    left = np.where(exact, right, np.maximum(right - 1, 0))
    alpha = np.zeros(right.shape, dtype=np.float64)
    different = right != left
    alpha[different] = (
        (output_wavelength[inside][different] - wavelength[left[different]])
        / (wavelength[right[different]] - wavelength[left[different]])
    )
    output_flux[positions] = flux[left] + alpha * (flux[right] - flux[left])

    variance = uncertainty**2
    # This conservative nearest-endpoint expression is intentionally the
    # one used by SpeXTool's mc_interpspec.pro, rather than the usual linear
    # interpolation variance formula.
    variance_sum = variance[left] + variance[right]
    from_left = variance[left] + alpha**2 * variance_sum
    from_right = variance[right] + (1.0 - alpha) ** 2 * variance_sum
    output_uncertainty[positions] = np.sqrt(np.minimum(from_left, from_right))
    output_mask[positions] = mask[left] | mask[right]
    output_mask[positions[exact]] = mask[right[exact]]
    return output_flux, output_uncertainty, output_mask


def _merge_pair(
    anchor: ExtractedOrder, addition: ExtractedOrder
) -> tuple[ExtractedOrder, tuple[float, float] | None]:
    w1, f1, e1, m1 = _trim_order(anchor)
    w2, f2, e2, m2 = _trim_order(addition)
    lower = max(w1[0], w2[0])
    upper = min(w1[-1], w2[-1])
    overlap = (float(lower), float(upper)) if lower <= upper else None

    i2f, i2e, i2m = _interpolate_order(w2, f2, e2, m2, w1)
    valid1 = np.isfinite(f1) & np.isfinite(e1) & (e1 > 0)
    valid2 = np.isfinite(i2f) & np.isfinite(i2e) & (i2e > 0)
    merged_flux = f1.copy()
    merged_uncertainty = e1.copy()
    merged_mask = m1.copy()
    both = valid1 & valid2
    if np.any(both):
        weight1 = 1.0 / e1[both] ** 2
        weight2 = 1.0 / i2e[both] ** 2
        merged_flux[both] = (f1[both] * weight1 + i2f[both] * weight2) / (
            weight1 + weight2
        )
        merged_uncertainty[both] = np.sqrt(1.0 / (weight1 + weight2))
        merged_mask[both] = m1[both] | i2m[both]
    only2 = ~valid1 & valid2
    merged_flux[only2] = i2f[only2]
    merged_uncertainty[only2] = i2e[only2]
    merged_mask[only2] = i2m[only2]

    outside = (w2 < w1[0]) | (w2 > w1[-1])
    output_wavelength = np.concatenate((w1, w2[outside]))
    output_flux = np.concatenate((merged_flux, f2[outside]))
    output_uncertainty = np.concatenate((merged_uncertainty, e2[outside]))
    output_mask = np.concatenate((merged_mask, m2[outside]))
    sorting = np.argsort(output_wavelength, kind="stable")
    return (
        ExtractedOrder(
            order=anchor.order,
            wavelength_micron=output_wavelength[sorting],
            flux=output_flux[sorting],
            uncertainty=output_uncertainty[sorting],
            mask=output_mask[sorting],
            trace_arcsec=np.full(output_wavelength.size, np.nan),
            background=np.full(output_wavelength.size, np.nan),
        ),
        overlap,
    )


def merge_orders(orders: tuple[ExtractedOrder, ...]) -> MergedSpectrum:
    """Merge adjacent orders following SpeXTool ``mc_mergespec.pro``.

    The first order is the initial wavelength-grid anchor. Each subsequent
    order is interpolated onto the accumulated grid in its overlap and
    inverse-variance averaged; non-overlapping pixels retain their native
    sampling.
    """

    if not orders:
        raise ValueError("at least one spectral order is required")
    merged = orders[0]
    overlaps: list[tuple[float, float] | None] = []
    for addition in orders[1:]:
        merged, overlap = _merge_pair(merged, addition)
        overlaps.append(overlap)
    return MergedSpectrum(
        wavelength_micron=merged.wavelength_micron,
        flux=merged.flux,
        uncertainty=merged.uncertainty,
        mask=merged.mask,
        orders=tuple(order.order for order in orders),
        overlap_ranges=tuple(overlaps),
    )


def write_merged_spectrum(product: MergedSpectrum, path: str | Path) -> Path:
    """Write a checksummed merged spectrum compatible with downstream FITS tools."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "MERGED"
    header["NORDERS"] = len(product.orders)
    header["ORDERS"] = ",".join(str(value) for value in product.orders)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="WAVELENGTH", format="D", unit="um", array=product.wavelength_micron
            ),
            fits.Column(name="FLUX", format="D", array=product.flux),
            fits.Column(name="UNCERTAINTY", format="D", array=product.uncertainty),
            fits.Column(name="MASK", format="I", array=product.mask),
        ],
        name="SPECTRUM",
    )
    fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
        output, overwrite=True, checksum=True
    )
    return output
