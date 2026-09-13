"""Combination and selected-order scaling of extracted spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from shuck.extraction.optimal import ExtractedExposure, ExtractedOrder
from shuck.statistics import robust_weighted_mean


@dataclass(frozen=True)
class CombinedObservationMetadata:
    """Observation metadata retained for telluric correction and provenance."""

    mode: str
    mean_mjd: float
    ra: str
    dec: str
    mean_airmass: float
    slit_width_arcsec: float
    plate_scale_arcsec_per_pixel: float


@dataclass(frozen=True)
class CombinedSpectrum:
    """Robustly combined multi-order spectrum for one control-file group."""

    orders: tuple[ExtractedOrder, ...]
    scale_order: int
    scale_factors: np.ndarray
    input_files: tuple[Path, ...]
    sigma_clip: float
    metadata: CombinedObservationMetadata | None = None


def _combine_observation_metadata(
    exposures: tuple[ExtractedExposure, ...],
) -> CombinedObservationMetadata | None:
    metadata = [exposure.metadata for exposure in exposures]
    plate_scales = [exposure.plate_scale_arcsec_per_pixel for exposure in exposures]
    if any(item is None for item in metadata) or any(value is None for value in plate_scales):
        return None
    complete = [item for item in metadata if item is not None]
    scales = np.array([value for value in plate_scales if value is not None])
    modes = {item.mode for item in complete}
    slit_widths = np.array([item.slit_width_arcsec for item in complete], dtype=np.float64)
    if len(modes) != 1:
        raise ValueError(f"cannot combine exposures from different modes: {modes}")
    if np.any(~np.isfinite(slit_widths)):
        raise ValueError("slit width metadata are required for telluric correction")
    if not np.allclose(slit_widths, slit_widths[0], rtol=0, atol=1e-6):
        raise ValueError("cannot combine exposures with different slit widths")
    if not np.allclose(scales, scales[0], rtol=0, atol=1e-12):
        raise ValueError("cannot combine exposures with different plate scales")
    return CombinedObservationMetadata(
        mode=modes.pop(),
        mean_mjd=float(np.mean([item.mjd_obs for item in complete])),
        ra=complete[0].ra,
        dec=complete[0].dec,
        mean_airmass=float(np.mean([item.airmass for item in complete])),
        slit_width_arcsec=float(slit_widths[0]),
        plate_scale_arcsec_per_pixel=float(scales[0]),
    )


def determine_scale_factors(
    spectra: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Scale spectra to their median following ``mc_getspecscale.pro``.

    The input should already be restricted to the central wavelength range
    accepted for scaling.
    """

    values = np.asarray(spectra, dtype=np.float64)
    valid = np.asarray(masks) == 0
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("scaling spectra and masks must have shape (nspectrum, npixel)")
    reference = np.nanmedian(np.where(valid, values, np.nan), axis=0)
    scales = np.full(values.shape[0], np.nan)
    for index in range(values.shape[0]):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = reference / values[index]
        good = valid[index] & np.isfinite(ratio)
        if np.any(good):
            scales[index] = np.median(ratio[good])
    if np.any(~np.isfinite(scales)):
        raise ValueError("could not determine a finite scale factor for every spectrum")
    return scales


def combine_exposures(
    exposures: tuple[ExtractedExposure, ...],
    *,
    scale_order: int,
    sigma_clip: float = 8.0,
) -> CombinedSpectrum:
    """Scale and robustly combine spectra as SpeXTool ``xcombspec.pro``."""

    if not exposures:
        raise ValueError("at least one extracted exposure is required")
    order_numbers = tuple(order.order for order in exposures[0].orders)
    if scale_order not in order_numbers:
        raise ValueError(f"scale order {scale_order} is absent from extracted spectra")
    for exposure in exposures[1:]:
        if tuple(order.order for order in exposure.orders) != order_numbers:
            raise ValueError("extracted exposures do not contain the same ordered order set")
    scale_index = order_numbers.index(scale_order)
    scale_spectra = np.stack([exposure.orders[scale_index].flux for exposure in exposures])
    scale_masks = np.stack([exposure.orders[scale_index].mask for exposure in exposures])
    n_pixels = scale_spectra.shape[1]
    central = slice(int(np.floor(0.1 * n_pixels)), int(np.ceil(0.9 * n_pixels)))
    scales = determine_scale_factors(scale_spectra[:, central], scale_masks[:, central])

    combined_orders: list[ExtractedOrder] = []
    for order_index, order_number in enumerate(order_numbers):
        reference_wavelength = exposures[0].orders[order_index].wavelength_micron
        for exposure in exposures[1:]:
            if not np.allclose(
                exposure.orders[order_index].wavelength_micron,
                reference_wavelength,
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(f"wavelength grids differ in order {order_number}")
        flux = np.stack(
            [
                exposure.orders[order_index].flux * scale
                for exposure, scale in zip(exposures, scales, strict=True)
            ]
        )
        variance = np.stack(
            [
                (exposure.orders[order_index].uncertainty * scale) ** 2
                for exposure, scale in zip(exposures, scales, strict=True)
            ]
        )
        masks = np.stack([exposure.orders[order_index].mask for exposure in exposures])
        combined = robust_weighted_mean(
            flux,
            variance,
            axis=0,
            sigma=sigma_clip,
            mask=masks == 0,
        )
        accepted_flags = np.where(combined.good, masks, 0).astype(np.uint16)
        output_mask = np.bitwise_or.reduce(accepted_flags, axis=0)
        failed = ~np.isfinite(combined.mean) | ~np.isfinite(combined.variance)
        output_mask[failed] |= np.uint16(8)
        combined_orders.append(
            ExtractedOrder(
                order=order_number,
                wavelength_micron=reference_wavelength.copy(),
                flux=combined.mean,
                uncertainty=np.sqrt(combined.variance),
                mask=output_mask,
                trace_arcsec=np.full_like(reference_wavelength, np.nan),
                background=np.full_like(reference_wavelength, np.nan),
            )
        )
    return CombinedSpectrum(
        orders=tuple(combined_orders),
        scale_order=scale_order,
        scale_factors=scales,
        input_files=tuple(exposure.source_path for exposure in exposures),
        sigma_clip=sigma_clip,
        metadata=_combine_observation_metadata(exposures),
    )


def write_combined_spectrum(
    product: CombinedSpectrum,
    path: str | Path,
    *,
    combination_id: str,
) -> Path:
    """Write a checksummed multi-order combined spectrum."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "COMBINED"
    header["COMBID"] = combination_id
    header["SCALEORD"] = product.scale_order
    header["SIGCLIP"] = product.sigma_clip
    header["NINPUTS"] = len(product.input_files)
    if product.metadata is not None:
        header["OBSMODE"] = product.metadata.mode
        header["AVE_MJD"] = product.metadata.mean_mjd
        header["AIRMASS"] = product.metadata.mean_airmass
        header["RA"] = product.metadata.ra
        header["DEC"] = product.metadata.dec
        header["SLTW_ARC"] = product.metadata.slit_width_arcsec
        header["PLTSCALE"] = product.metadata.plate_scale_arcsec_per_pixel
    for source, scale in zip(product.input_files, product.scale_factors, strict=True):
        header.add_history(f"INPUT {source.name} SCALE {scale:.10g}")
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]
    for order in product.orders:
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="WAVELENGTH", format="D", unit="um", array=order.wavelength_micron
                    ),
                    fits.Column(name="FLUX", format="D", unit="DN/s", array=order.flux),
                    fits.Column(
                        name="UNCERTAINTY",
                        format="D",
                        unit="DN/s",
                        array=order.uncertainty,
                    ),
                    fits.Column(name="MASK", format="I", array=order.mask),
                ],
                name=f"ORDER{order.order:03d}",
            )
        )
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output
