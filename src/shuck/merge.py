"""Order merging for telluric-corrected spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from shuck.combine import CombinedObservationMetadata
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
    science_group: str | None = None
    standard_group: str | None = None
    science_files: tuple[Path, ...] = ()
    standard_files: tuple[Path, ...] = ()
    science_airmass: float | None = None
    standard_airmass: float | None = None
    observation_metadata: CombinedObservationMetadata | None = None


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
    alpha[different] = (output_wavelength[inside][different] - wavelength[left[different]]) / (
        wavelength[right[different]] - wavelength[left[different]]
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
    minimum1, maximum1 = w1[0], w1[-1]
    minimum2, maximum2 = w2[0], w2[-1]
    if minimum2 > maximum1:
        output_wavelength = np.concatenate((w1, w2))
        output_flux = np.concatenate((f1[:-1], [np.nan, np.nan], f2[1:]))
        output_uncertainty = np.concatenate((e1[:-1], [np.nan, np.nan], e2[1:]))
        output_mask = np.concatenate((m1[:-1], [0, 0], m2[1:])).astype(np.uint16)
        return _merged_order(
            anchor.order, output_wavelength, output_flux, output_uncertainty, output_mask
        ), None
    if minimum1 > maximum2:
        output_wavelength = np.concatenate((w2, w1))
        output_flux = np.concatenate((f2[:-1], [np.nan, np.nan], f1[1:]))
        output_uncertainty = np.concatenate((e2[:-1], [np.nan, np.nan], e1[1:]))
        output_mask = np.concatenate((m2[:-1], [0, 0], m1[1:])).astype(np.uint16)
        return _merged_order(
            anchor.order, output_wavelength, output_flux, output_uncertainty, output_mask
        ), None

    interpolation_wavelength = w2
    interpolation_flux = f2
    interpolation_uncertainty = e2
    interpolation_mask = m2
    position = "left" if minimum2 < minimum1 else "right"
    if minimum2 >= minimum1 and maximum2 <= maximum1:
        position = "inside"
        interior = (w2 > minimum1) & (w2 < maximum1)
        interpolation_wavelength = w2[interior]
        interpolation_flux = f2[interior]
        interpolation_uncertainty = e2[interior]
        interpolation_mask = m2[interior]
        if interpolation_wavelength.size < 2:
            raise ValueError("contained order has fewer than two interior wavelength samples")

    i2f, i2e, i2m = _interpolate_order(
        interpolation_wavelength,
        interpolation_flux,
        interpolation_uncertainty,
        interpolation_mask,
        w1,
    )
    valid1 = np.isfinite(f1) & np.isfinite(e1) & (e1 > 0)
    valid2 = np.isfinite(i2f) & np.isfinite(i2e) & (i2e > 0)
    middle_flux = f1.copy()
    middle_uncertainty = e1.copy()
    middle_mask = m1.copy()
    both = valid1 & valid2
    if np.any(both):
        weight1 = 1.0 / e1[both] ** 2
        weight2 = 1.0 / i2e[both] ** 2
        middle_flux[both] = (f1[both] * weight1 + i2f[both] * weight2) / (weight1 + weight2)
        middle_uncertainty[both] = np.sqrt(1.0 / (weight1 + weight2))
        middle_mask[both] = m1[both] | i2m[both]
    only2 = ~valid1 & valid2
    middle_flux[only2] = i2f[only2]
    middle_uncertainty[only2] = i2e[only2]
    middle_mask[only2] = i2m[only2]

    finite_interpolation = np.flatnonzero(np.isfinite(i2f))
    if finite_interpolation.size == 0:
        raise ValueError("overlapping orders have no finite interpolated samples")
    if position == "right":
        middle_indices = np.arange(finite_interpolation[0], w1.size)
        left_indices = np.flatnonzero(w1 < minimum2)
        right_candidates = np.flatnonzero(w2 >= np.max(w1[middle_indices]))
        right_indices = right_candidates[1:]
        segments = (
            (w1[left_indices], f1[left_indices], e1[left_indices], m1[left_indices]),
            (
                w1[middle_indices],
                middle_flux[middle_indices],
                middle_uncertainty[middle_indices],
                middle_mask[middle_indices],
            ),
            (w2[right_indices], f2[right_indices], e2[right_indices], m2[right_indices]),
        )
        overlap = (float(minimum2), float(np.max(w1[middle_indices])))
    elif position == "left":
        middle_indices = np.arange(0, finite_interpolation[-1] + 1)
        left_candidates = np.flatnonzero(w2 <= minimum1)
        left_indices = left_candidates[:-1]
        right_indices = np.flatnonzero(w1 > np.max(w1[middle_indices]))
        segments = (
            (w2[left_indices], f2[left_indices], e2[left_indices], m2[left_indices]),
            (
                w1[middle_indices],
                middle_flux[middle_indices],
                middle_uncertainty[middle_indices],
                middle_mask[middle_indices],
            ),
            (w1[right_indices], f1[right_indices], e1[right_indices], m1[right_indices]),
        )
        overlap = (float(minimum1), float(np.max(w1[middle_indices])))
    else:
        middle_indices = np.arange(finite_interpolation[0], finite_interpolation[-1] + 1)
        left_candidates = np.flatnonzero(w1 <= minimum2)
        left_indices = left_candidates[:-1]
        right_indices = np.flatnonzero(w1 > maximum2)
        segments = (
            (w1[left_indices], f1[left_indices], e1[left_indices], m1[left_indices]),
            (
                w1[middle_indices],
                middle_flux[middle_indices],
                middle_uncertainty[middle_indices],
                middle_mask[middle_indices],
            ),
            (w1[right_indices], f1[right_indices], e1[right_indices], m1[right_indices]),
        )
        overlap = (float(minimum1), float(maximum2))

    output_wavelength = np.concatenate([segment[0] for segment in segments])
    output_flux = np.concatenate([segment[1] for segment in segments])
    output_uncertainty = np.concatenate([segment[2] for segment in segments])
    output_mask = np.concatenate([segment[3] for segment in segments])
    return _merged_order(
        anchor.order,
        output_wavelength,
        output_flux,
        output_uncertainty,
        output_mask,
    ), overlap


def _merged_order(
    order: int,
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
) -> ExtractedOrder:
    """Package one intermediate ``mc_mergespec.pro``-equivalent result."""

    return ExtractedOrder(
        order=order,
        wavelength_micron=np.asarray(wavelength, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        mask=np.asarray(mask, dtype=np.uint16),
        trace_arcsec=np.full(wavelength.size, np.nan),
        background=np.full(wavelength.size, np.nan),
    )


def merge_orders(
    orders: tuple[ExtractedOrder, ...],
    *,
    science_group: str | None = None,
    standard_group: str | None = None,
    science_files: tuple[Path, ...] = (),
    standard_files: tuple[Path, ...] = (),
    science_airmass: float | None = None,
    standard_airmass: float | None = None,
    observation_metadata: CombinedObservationMetadata | None = None,
) -> MergedSpectrum:
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
        science_group=science_group,
        standard_group=standard_group,
        science_files=science_files,
        standard_files=standard_files,
        science_airmass=science_airmass,
        standard_airmass=standard_airmass,
        observation_metadata=observation_metadata,
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
    if product.science_group is not None:
        header["SCIGRP"] = product.science_group
    if product.standard_group is not None:
        header["STDGRP"] = product.standard_group
    if product.science_airmass is not None:
        header["AIRMASS"] = product.science_airmass
    if product.standard_airmass is not None:
        header["STDAMASS"] = product.standard_airmass
    if product.science_airmass is not None and product.standard_airmass is not None:
        header["AMDIFF"] = product.standard_airmass - product.science_airmass
    if product.observation_metadata is not None:
        header["OBSMODE"] = product.observation_metadata.mode
        header["AVE_MJD"] = product.observation_metadata.mean_mjd
        header["RA"] = product.observation_metadata.ra
        header["DEC"] = product.observation_metadata.dec
        header["SLTW_ARC"] = product.observation_metadata.slit_width_arcsec
        header["PLTSCALE"] = product.observation_metadata.plate_scale_arcsec_per_pixel
    for source in product.science_files:
        header.add_history(f"SCIENCE {source.name}")
    for source in product.standard_files:
        header.add_history(f"STANDARD {source.name}")
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVELENGTH", format="D", unit="um", array=product.wavelength_micron),
            fits.Column(
                name="FLUX",
                format="D",
                unit="erg s-1 cm-2 Angstrom-1",
                array=product.flux,
            ),
            fits.Column(
                name="UNCERTAINTY",
                format="D",
                unit="erg s-1 cm-2 Angstrom-1",
                array=product.uncertainty,
            ),
            fits.Column(name="MASK", format="I", array=product.mask),
        ],
        name="SPECTRUM",
    )
    fits.HDUList([fits.PrimaryHDU(header=header), table]).writeto(
        output, overwrite=True, checksum=True
    )
    return output
