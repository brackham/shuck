"""Optimal point-source extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from shuck.calibration.flat import _robust_polynomial_fit
from shuck.calibration.rectify import RectifiedOrder
from shuck.extraction.preprocess import PreprocessedExposure
from shuck.extraction.profile import (
    ApertureLocation,
    SpatialProfile,
    find_apertures,
    make_spatial_profiles,
)
from shuck.extraction.trace import SpectralTrace, trace_orders
from shuck.io import IShellRawMetadata
from shuck.statistics import robust_weighted_mean

EXTRACTION_FAILURE_BIT = np.uint16(8)


@dataclass(frozen=True)
class ExtractedOrder:
    """One optimally extracted point-source spectral order."""

    order: int
    wavelength_micron: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray
    trace_arcsec: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class ExtractedExposure:
    """All extracted orders and diagnostic models for one exposure."""

    orders: tuple[ExtractedOrder, ...]
    profiles: tuple[SpatialProfile, ...]
    apertures: tuple[ApertureLocation, ...]
    traces: tuple[SpectralTrace, ...]
    source_path: Path
    metadata: IShellRawMetadata | None = None
    plate_scale_arcsec_per_pixel: float | None = None


@dataclass(frozen=True)
class ExtractionModel:
    """Group-level spatial profiles, apertures, and wavelength traces."""

    profiles: tuple[SpatialProfile, ...]
    apertures: tuple[ApertureLocation, ...]
    traces: tuple[SpectralTrace, ...]


def _profile_model(
    order: RectifiedOrder,
    profile: SpatialProfile,
    trace: SpectralTrace,
    aperture_radius: float,
) -> np.ndarray:
    """Generate the wavelength-dependent model used by ``mc_mkspatmodel.pro``."""

    base = profile.normalized_flux
    trace_positions = trace.position(order.wavelength_micron)
    median_trace = float(np.nanmedian(trace_positions))
    aperture = np.abs(order.spatial_arcsec - median_trace) <= aperture_radius
    valid_base = np.isfinite(base) & (base != 0) & aperture
    good_image = (order.mask == 0) & np.isfinite(order.image)
    column_background = np.nanmedian(np.where(good_image, order.image, np.nan), axis=0)
    background_subtracted = order.image - column_background[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = background_subtracted / base[:, None]
    ratio[~valid_base, :] = np.nan
    ratio[(order.mask != 0)] = np.nan
    normalization = np.nanmedian(ratio, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = background_subtracted / normalization[None, :]
    model = np.full_like(order.image, np.nan, dtype=np.float64)
    for row in range(order.image.shape[0]):
        good = (order.mask[row] == 0) & np.isfinite(normalized[row])
        if np.count_nonzero(good) < 3:
            continue
        coefficients = _robust_polynomial_fit(
            order.wavelength_micron[good],
            normalized[row, good],
            2,
            threshold=3.5,
            fractional_sigma_change=0.01,
        )
        model[row] = np.polynomial.polynomial.polyval(order.wavelength_micron, coefficients)
    return model


def _scaled_profile_good(
    model: np.ndarray,
    data: np.ndarray,
    variance: np.ndarray,
    initially_good: np.ndarray,
    *,
    threshold: float = 7.0,
) -> np.ndarray:
    good = initially_good & np.isfinite(model) & np.isfinite(data) & np.isfinite(variance)
    good &= variance > 0
    design = np.column_stack((np.ones(len(model)), model))
    for _ in range(5):
        if np.count_nonzero(good) < 2:
            return good
        weighted_design = design[good] / np.sqrt(variance[good])[:, None]
        weighted_data = data[good] / np.sqrt(variance[good])
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_data, rcond=None)
        residual = data - design @ coefficients
        median = np.median(residual[good])
        mad = 1.482 * np.median(np.abs(residual[good] - median))
        noise_scale = float(np.median(np.sqrt(variance[good])))
        scale = max(mad, noise_scale)
        if scale == 0:
            break
        updated = good & (np.abs(residual - median) <= threshold * scale)
        if np.array_equal(updated, good):
            break
        good = updated
    return good


def extract_order_optimal(
    order: RectifiedOrder,
    profile: SpatialProfile,
    aperture: ApertureLocation,
    trace: SpectralTrace,
    *,
    psf_radius: float = 1.0,
    aperture_radius: float = 1.0,
    background_start: float = 1.1,
    background_width: float = 2.0,
) -> ExtractedOrder:
    """Optimally extract one order following SpeXTool ``mc_extpsspec.pro``."""

    if not aperture.valid or not trace.valid:
        size = len(order.wavelength_micron)
        return ExtractedOrder(
            order=order.order,
            wavelength_micron=order.wavelength_micron.copy(),
            flux=np.full(size, np.nan),
            uncertainty=np.full(size, np.nan),
            mask=np.full(size, EXTRACTION_FAILURE_BIT, dtype=np.uint16),
            trace_arcsec=np.full(size, np.nan),
            background=np.full(size, np.nan),
        )
    model = _profile_model(order, profile, trace, aperture_radius)
    positions = trace.position(order.wavelength_micron)
    n_columns = order.image.shape[1]
    flux = np.full(n_columns, np.nan)
    uncertainty = np.full(n_columns, np.nan)
    output_mask = np.zeros(n_columns, dtype=np.uint16)
    background = np.full(n_columns, np.nan)
    for column in range(n_columns):
        offset = np.abs(order.spatial_arcsec - positions[column])
        background_pixels = (
            (offset >= background_start)
            & (offset <= background_start + background_width)
            & (order.mask[:, column] == 0)
        )
        if np.count_nonzero(background_pixels) < 2:
            output_mask[column] |= EXTRACTION_FAILURE_BIT
            continue
        background_fit = robust_weighted_mean(
            order.image[background_pixels, column],
            order.variance[background_pixels, column],
            sigma=4.0,
        )
        background[column] = float(background_fit.mean)
        background_variance = float(background_fit.variance)
        data = order.image[:, column] - background[column]
        variance = order.variance[:, column] + background_variance
        good = _scaled_profile_good(
            model[:, column],
            data,
            variance,
            order.mask[:, column] == 0,
        )
        positive_profile = aperture.sign * model[:, column]
        positive_profile = np.where(positive_profile > 0, positive_profile, 0.0)
        normalization = np.nansum(positive_profile)
        if not np.isfinite(normalization) or normalization <= 0:
            output_mask[column] |= EXTRACTION_FAILURE_BIT
            continue
        aperture_profile = aperture.sign * positive_profile / normalization
        aperture_pixels = (
            (offset <= min(aperture_radius, psf_radius))
            & good
            & np.isfinite(aperture_profile)
            & (aperture_profile != 0)
        )
        if not np.any(aperture_pixels):
            output_mask[column] |= EXTRACTION_FAILURE_BIT
            continue
        values = data[aperture_pixels] / aperture_profile[aperture_pixels]
        values_variance = variance[aperture_pixels] / aperture_profile[aperture_pixels] ** 2
        combined = robust_weighted_mean(values, values_variance, sigma=8.0)
        flux[column] = float(combined.mean)
        uncertainty[column] = float(np.sqrt(combined.variance))
        footprint = offset <= aperture_radius
        if np.any(footprint):
            output_mask[column] |= np.bitwise_or.reduce(order.mask[footprint, column])
    return ExtractedOrder(
        order=order.order,
        wavelength_micron=order.wavelength_micron.copy(),
        flux=flux,
        uncertainty=uncertainty,
        mask=output_mask,
        trace_arcsec=positions,
        background=background,
    )


def extract_exposure(
    exposure: PreprocessedExposure,
    *,
    model: ExtractionModel | None = None,
    trace_degree: int = 2,
    psf_radius: float = 1.0,
    aperture_radius: float = 1.0,
    background_start: float = 1.1,
    background_width: float = 2.0,
) -> ExtractedExposure:
    """Run the automatic one-aperture SpeXTool point-source extraction path."""

    if model is None:
        model = derive_extraction_model(exposure, trace_degree=trace_degree)
    profiles = model.profiles
    apertures = model.apertures
    traces = model.traces
    profile_by_order = {profile.order: profile for profile in profiles}
    aperture_by_order = {aperture.order: aperture for aperture in apertures}
    trace_by_order = {trace.order: trace for trace in traces}
    extracted = tuple(
        extract_order_optimal(
            order,
            profile_by_order[order.order],
            aperture_by_order[order.order],
            trace_by_order[order.order],
            psf_radius=psf_radius,
            aperture_radius=aperture_radius,
            background_start=background_start,
            background_width=background_width,
        )
        for order in exposure.orders
    )
    return ExtractedExposure(
        orders=extracted,
        profiles=profiles,
        apertures=apertures,
        traces=traces,
        source_path=exposure.source_path,
        metadata=exposure.metadata,
        plate_scale_arcsec_per_pixel=exposure.plate_scale_arcsec_per_pixel,
    )


def derive_extraction_model(
    exposure: PreprocessedExposure,
    *,
    trace_degree: int = 2,
) -> ExtractionModel:
    """Derive the automatic SpeXTool aperture and trace model for a group."""

    profiles = make_spatial_profiles(exposure.orders)
    apertures = find_apertures(profiles)
    traces = trace_orders(exposure.orders, apertures, degree=trace_degree)
    return ExtractionModel(profiles=profiles, apertures=apertures, traces=traces)


def write_extracted_exposure(product: ExtractedExposure, path: str | Path) -> Path:
    """Write extracted orders and diagnostic vectors to a checksummed FITS product."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.Header()
    primary["SHUCKVER"] = "0.1.0.dev0"
    primary["STAGE"] = "EXTRACTED"
    primary["NORDERS"] = len(product.orders)
    if product.metadata is not None:
        primary["OBSMODE"] = product.metadata.mode
        primary["MJD_OBS"] = product.metadata.mjd_obs
        primary["AIRMASS"] = product.metadata.airmass
        primary["RA"] = product.metadata.ra
        primary["DEC"] = product.metadata.dec
        if product.metadata.slit_width_arcsec is not None:
            primary["SLTW_ARC"] = product.metadata.slit_width_arcsec
    if product.plate_scale_arcsec_per_pixel is not None:
        primary["PLTSCALE"] = product.plate_scale_arcsec_per_pixel
    primary.add_history(f"INPUT {product.source_path.name}")
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=primary)]
    aperture_by_order = {item.order: item for item in product.apertures}
    trace_by_order = {item.order: item for item in product.traces}
    for order in product.orders:
        aperture = aperture_by_order[order.order]
        trace = trace_by_order[order.order]
        columns = [
            fits.Column(name="WAVELENGTH", format="D", unit="um", array=order.wavelength_micron),
            fits.Column(name="FLUX", format="D", unit="DN/s", array=order.flux),
            fits.Column(name="UNCERTAINTY", format="D", unit="DN/s", array=order.uncertainty),
            fits.Column(name="MASK", format="I", array=order.mask),
            fits.Column(name="TRACE", format="D", unit="arcsec", array=order.trace_arcsec),
            fits.Column(name="BACKGROUND", format="D", unit="DN/s", array=order.background),
        ]
        extension = fits.BinTableHDU.from_columns(columns, name=f"ORDER{order.order:03d}")
        extension.header["APPOS"] = aperture.position_arcsec
        extension.header["APSIGN"] = aperture.sign
        extension.header["APFWHM"] = aperture.fwhm_arcsec
        extension.header["TRCRMS"] = trace.rms_arcsec
        for index, coefficient in enumerate(trace.coefficients):
            extension.header[f"TRC_C{index}"] = coefficient
        hdus.append(extension)
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output


def read_extracted_exposure(
    path: str | Path,
    *,
    source_path: str | Path,
    metadata: IShellRawMetadata,
    plate_scale_arcsec_per_pixel: float,
) -> ExtractedExposure:
    """Read a cached shuck extraction for combination without recomputing images."""

    input_path = Path(path)
    orders: list[ExtractedOrder] = []
    with fits.open(input_path, mode="readonly", memmap=False, checksum=True) as hdus:
        invalid_checksums = [
            hdu.name for hdu in hdus if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1
        ]
        if invalid_checksums:
            raise ValueError(
                f"cached extraction has invalid FITS checksums in {invalid_checksums}: {input_path}"
            )
        if hdus[0].header.get("STAGE") != "EXTRACTED":
            raise ValueError(f"cached product is not an extracted spectrum: {input_path}")
        for hdu in hdus[1:]:
            if not hdu.name.startswith("ORDER"):
                continue
            number = int(hdu.name.removeprefix("ORDER"))
            orders.append(
                ExtractedOrder(
                    order=number,
                    wavelength_micron=np.asarray(hdu.data["WAVELENGTH"], dtype=np.float64),
                    flux=np.asarray(hdu.data["FLUX"], dtype=np.float64),
                    uncertainty=np.asarray(hdu.data["UNCERTAINTY"], dtype=np.float64),
                    mask=np.asarray(hdu.data["MASK"], dtype=np.uint16),
                    trace_arcsec=np.asarray(hdu.data["TRACE"], dtype=np.float64),
                    background=np.asarray(hdu.data["BACKGROUND"], dtype=np.float64),
                )
            )
    if not orders:
        raise ValueError(f"cached extraction contains no spectral orders: {input_path}")
    return ExtractedExposure(
        orders=tuple(orders),
        profiles=(),
        apertures=(),
        traces=(),
        source_path=Path(source_path),
        metadata=metadata,
        plate_scale_arcsec_per_pixel=plate_scale_arcsec_per_pixel,
    )
