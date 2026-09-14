"""A0V/Vega telluric correction for the supported iSHELL IP path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import FK4, FK5, SkyCoord, get_body_barycentric_posvel
from astropy.io import fits
from astropy.time import Time
from scipy.io import readsav
from scipy.signal import convolve
from scipy.special import erf

from shuck.combine import CombinedObservationMetadata, CombinedSpectrum
from shuck.extraction.optimal import ExtractedOrder

SPEED_OF_LIGHT_KMS = 299_792.458
VEGA_V_MAGNITUDE = 0.03
VEGA_B_MINUS_V = 0.0
DEFAULT_RV = 3.1
TELLURIC_INVALID_BIT = np.uint16(8)


@dataclass(frozen=True)
class VegaModel:
    """High-resolution Vega spectrum and its two continuum estimates."""

    wavelength_micron: np.ndarray
    flux: np.ndarray
    line_continuum: np.ndarray
    broad_continuum: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class TelluricCorrection:
    """Multiplicative telluric/response correction derived from an A0V star."""

    orders: tuple[ExtractedOrder, ...]
    vega_orders: tuple[ExtractedOrder, ...]
    standard_orders: tuple[ExtractedOrder, ...]
    standard_group: str
    standard_files: tuple[Path, ...]
    b_magnitude: float
    v_magnitude: float
    catalog_rv_kms: float
    earth_lsr_velocity_kms: float
    applied_velocity_shift_kms: float
    standard_airmass: float
    slit_width_arcsec: float
    ip_coefficients: np.ndarray
    ip_coefficients_path: Path
    vega_model_path: Path


@dataclass(frozen=True)
class TelluricCorrectedSpectrum:
    """One combined science spectrum after telluric and flux correction."""

    orders: tuple[ExtractedOrder, ...]
    science_group: str
    standard_group: str
    science_files: tuple[Path, ...]
    standard_files: tuple[Path, ...]
    science_airmass: float | None
    standard_airmass: float | None
    science_metadata: CombinedObservationMetadata | None


def load_vega_model(spextool_directory: str | Path) -> VegaModel:
    """Load SpeXTool's R=500,000 Vega model used by ``xtellcor.pro``.

    The pipeline passes the bundled package-data root; an explicit root can
    be supplied for developer comparisons.
    """

    path = Path(spextool_directory).expanduser().resolve() / "data" / "vega500000.sav"
    if not path.is_file():
        raise FileNotFoundError(f"SpeXTool Vega model does not exist: {path}")
    values = readsav(path, python_dict=True)

    def array(name: str) -> np.ndarray:
        result = np.asarray(values[name], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Vega model field {name} contains non-finite values")
        return result

    model = VegaModel(
        wavelength_micron=array("wvin"),
        flux=array("fvin"),
        line_continuum=array("fcvin"),
        broad_continuum=array("fc2vin"),
        source_path=path,
    )
    sizes = {
        model.wavelength_micron.size,
        model.flux.size,
        model.line_continuum.size,
        model.broad_continuum.size,
    }
    if len(sizes) != 1 or model.wavelength_micron.size < 2:
        raise ValueError("Vega model arrays do not have a common nontrivial length")
    if np.any(np.diff(model.wavelength_micron) <= 0):
        raise ValueError("Vega model wavelength grid is not strictly increasing")
    return model


def earth_lsr_velocity_kms(mjd: float, ra: str, dec: str) -> float:
    """Return SpeXTool ``mc_earthvelocity.pro``'s projected LSR velocity.

    SpeXTool evaluates its legacy barycentric ephemeris at the integer hour
    of the combined standard observation and adds the 20 km/s standard solar
    motion toward the B1900 apex at (18h, +30 degrees). Astropy supplies the
    modern Earth ephemeris and reference-frame transformation here.
    """

    observed = Time(mjd, format="mjd", scale="utc")
    day = observed.to_datetime()
    integer_hour = Time(
        f"{day.year:04d}-{day.month:02d}-{day.day:02d}T{day.hour:02d}:00:00",
        scale="utc",
    )
    coordinate = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame=FK5(equinox="J2000"))
    _, earth_velocity = get_body_barycentric_posvel("earth", integer_hour)
    orbital = float(
        np.dot(
            earth_velocity.xyz.to_value(u.km / u.s),
            coordinate.cartesian.xyz.value,
        )
    )
    solar_apex = SkyCoord(
        18 * u.hourangle,
        30 * u.deg,
        frame=FK4(equinox="B1900"),
    ).transform_to(FK5(equinox="J2000"))
    solar = float(20.0 * np.dot(solar_apex.cartesian.xyz.value, coordinate.cartesian.xyz.value))
    return orbital + solar


def instrument_profile(x_pixels: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the iSHELL slit-convolved Gaussian in ``mc_instrprof.pro``."""

    x = np.asarray(x_pixels, dtype=np.float64)
    parameters = np.asarray(coefficients, dtype=np.float64)
    if parameters.shape != (3,) or parameters[2] <= 0:
        raise ValueError("instrument-profile coefficients must be [center, half_width, sigma]")
    profile = erf((x + parameters[1] - parameters[0]) / parameters[2]) - erf(
        (x - parameters[1] - parameters[0]) / parameters[2]
    )
    normalization = np.sum(profile)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("instrument profile could not be normalized")
    return profile / normalization


def load_ip_coefficients(spextool_directory: str | Path, slit_width_arcsec: float) -> np.ndarray:
    """Select an iSHELL precomputed IP row as done by ``xtellcor.pro``."""

    path = (
        Path(spextool_directory).expanduser().resolve()
        / "instruments"
        / "ishell"
        / "data"
        / "IP_coefficients.dat"
    )
    table = np.loadtxt(path, comments="#", dtype=np.float64)
    matches = np.flatnonzero(np.isclose(table[:, 0], slit_width_arcsec, rtol=0, atol=1e-6))
    if matches.size != 1:
        raise ValueError(
            f"slit width {slit_width_arcsec:g} arcsec has no unique iSHELL IP calibration"
        )
    return table[matches[0], 1:].copy()


def _redden_ccm(wavelength_micron: np.ndarray, flux: np.ndarray, ebmv: float) -> np.ndarray:
    """Apply the near-IR CCM law selected by SpeXTool ``mc_redden.pro``."""

    if ebmv <= 0:
        return flux.copy()
    inverse_micron = 1.0 / wavelength_micron
    a = 0.574 * inverse_micron**1.61
    b = -0.527 * inverse_micron**1.61
    extinction_magnitude = ebmv * (DEFAULT_RV * a + b)
    return flux * 10.0 ** (-0.4 * extinction_magnitude)


def _fixed_dispersion_kernel(
    standard_wavelength: np.ndarray,
    vega_wavelength: np.ndarray,
    slit_width_pixels: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    dispersion = float(np.nanmean(np.diff(standard_wavelength)))
    covered = (vega_wavelength > np.nanmin(standard_wavelength)) & (
        vega_wavelength < np.nanmax(standard_wavelength)
    )
    if np.count_nonzero(covered) < 2 or not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("insufficient wavelength coverage for an IP kernel")
    vega_dispersion = float(np.mean(np.diff(vega_wavelength[covered])))
    kernel_width_pixels = dispersion * slit_width_pixels / vega_dispersion
    kernel_size = int(np.floor(7.0 * kernel_width_pixels + 0.5))
    kernel_size = max(kernel_size, 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_vega_wavelength = (
        np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    ) * vega_dispersion
    return instrument_profile(kernel_vega_wavelength / dispersion, coefficients)


def _convolved_vega(
    standard_wavelength: np.ndarray,
    vega: VegaModel,
    kernel: np.ndarray,
    velocity_shift_kms: float,
) -> np.ndarray:
    shifted_wavelength = vega.wavelength_micron * (1.0 + velocity_shift_kms / SPEED_OF_LIGHT_KMS)
    covered = (shifted_wavelength >= np.nanmin(standard_wavelength)) & (
        shifted_wavelength <= np.nanmax(standard_wavelength)
    )
    positions = np.flatnonzero(covered)
    if positions.size == 0:
        raise ValueError("Vega model does not cover standard wavelength grid")
    padding = kernel.size
    start = max(0, positions[0] - padding)
    stop = min(shifted_wavelength.size, positions[-1] + padding + 1)
    selection = slice(start, stop)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_line = vega.flux[selection] / vega.line_continuum[selection] - 1.0
        normalized_continuum = (
            vega.line_continuum[selection] / vega.broad_continuum[selection] - 1.0
        )
    convolved_line = convolve(normalized_line, kernel, mode="same", method="direct") + 1.0
    convolved_continuum = convolve(normalized_continuum, kernel, mode="same", method="direct") + 1.0
    convolved_flux = convolved_line * convolved_continuum * vega.broad_continuum[selection]
    return np.interp(
        standard_wavelength,
        shifted_wavelength[selection],
        convolved_flux,
        left=np.nan,
        right=np.nan,
    )


def _fill_correction(
    wavelength: np.ndarray,
    correction: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(correction)
        & np.isfinite(uncertainty)
        & (correction > 0)
        & (uncertainty >= 0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("fewer than two valid telluric-correction samples remain")
    filled = np.interp(wavelength, wavelength[valid], correction[valid], left=np.nan, right=np.nan)
    filled_uncertainty = np.interp(
        wavelength,
        wavelength[valid],
        uncertainty[valid],
        left=np.nan,
        right=np.nan,
    )
    return filled, filled_uncertainty


def build_telluric_correction(
    standard: CombinedSpectrum,
    *,
    standard_group: str,
    b_magnitude: float,
    v_magnitude: float,
    radial_velocity_kms: float,
    spextool_directory: str | Path,
) -> TelluricCorrection:
    """Construct an A0V correction via SpeXTool ``mc_mktellspec.pro``.

    This is the supported fixed-dispersion, precomputed-iSHELL-IP path. The
    optional hydrogen-line scaling and residual shift-fitting paths are
    deliberately omitted by the v0.1 specification.
    """

    if standard.metadata is None:
        raise ValueError("combined standard is missing observation metadata")
    metadata = standard.metadata
    slit_width_pixels = metadata.slit_width_arcsec / metadata.plate_scale_arcsec_per_pixel
    vega = load_vega_model(spextool_directory)
    spextool_root = Path(spextool_directory).expanduser().resolve()
    ip_coefficients_path = spextool_root / "instruments" / "ishell" / "data" / "IP_coefficients.dat"
    coefficients = load_ip_coefficients(spextool_root, metadata.slit_width_arcsec)
    earth_velocity = earth_lsr_velocity_kms(metadata.mean_mjd, metadata.ra, metadata.dec)
    velocity_shift = radial_velocity_kms - earth_velocity
    magnitude_scale = 10.0 ** (-0.4 * (v_magnitude - VEGA_V_MAGNITUDE))
    ebmv = max(b_magnitude - v_magnitude - VEGA_B_MINUS_V, 0.0)
    av = DEFAULT_RV * ebmv

    correction_orders: list[ExtractedOrder] = []
    vega_orders: list[ExtractedOrder] = []
    for order in standard.orders:
        wavelength = order.wavelength_micron
        kernel = _fixed_dispersion_kernel(
            wavelength,
            vega.wavelength_micron,
            slit_width_pixels,
            coefficients,
        )
        model_flux = _convolved_vega(wavelength, vega, kernel, velocity_shift)
        model_flux = _redden_ccm(wavelength, model_flux, ebmv)
        model_flux *= magnitude_scale * 10.0 ** (0.4 * av)
        usable = (
            (order.mask == 0)
            & np.isfinite(order.flux)
            & np.isfinite(order.uncertainty)
            & (order.flux > 0)
        )
        correction = np.full(wavelength.shape, np.nan)
        correction_uncertainty = np.full(wavelength.shape, np.nan)
        correction[usable] = model_flux[usable] / order.flux[usable]
        correction_uncertainty[usable] = np.abs(
            model_flux[usable] / order.flux[usable] ** 2 * order.uncertainty[usable]
        )
        correction, correction_uncertainty = _fill_correction(
            wavelength, correction, correction_uncertainty
        )
        invalid = ~np.isfinite(correction) | ~np.isfinite(correction_uncertainty)
        output_mask = order.mask.copy()
        output_mask[invalid] |= TELLURIC_INVALID_BIT
        correction_orders.append(
            ExtractedOrder(
                order=order.order,
                wavelength_micron=wavelength.copy(),
                flux=correction,
                uncertainty=correction_uncertainty,
                mask=output_mask,
                trace_arcsec=np.full(wavelength.shape, np.nan),
                background=np.full(wavelength.shape, np.nan),
            )
        )
        vega_orders.append(
            ExtractedOrder(
                order=order.order,
                wavelength_micron=wavelength.copy(),
                flux=model_flux,
                uncertainty=np.zeros(wavelength.shape),
                mask=np.where(np.isfinite(model_flux), 0, TELLURIC_INVALID_BIT).astype(np.uint16),
                trace_arcsec=np.full(wavelength.shape, np.nan),
                background=np.full(wavelength.shape, np.nan),
            )
        )
    return TelluricCorrection(
        orders=tuple(correction_orders),
        vega_orders=tuple(vega_orders),
        standard_orders=standard.orders,
        standard_group=standard_group,
        standard_files=standard.input_files,
        b_magnitude=b_magnitude,
        v_magnitude=v_magnitude,
        catalog_rv_kms=radial_velocity_kms,
        earth_lsr_velocity_kms=earth_velocity,
        applied_velocity_shift_kms=velocity_shift,
        standard_airmass=metadata.mean_airmass,
        slit_width_arcsec=metadata.slit_width_arcsec,
        ip_coefficients=coefficients,
        ip_coefficients_path=ip_coefficients_path,
        vega_model_path=vega.source_path,
    )


def _resample_correction(
    correction: ExtractedOrder, wavelength: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_wavelength = correction.wavelength_micron
    order = np.argsort(source_wavelength)
    source_wavelength = source_wavelength[order]
    source_flux = correction.flux[order]
    source_uncertainty = correction.uncertainty[order]
    source_mask = correction.mask[order]
    right = np.searchsorted(source_wavelength, wavelength, side="left")
    inside = (wavelength >= source_wavelength[0]) & (wavelength <= source_wavelength[-1])
    right = np.clip(right, 0, source_wavelength.size - 1)
    left = np.maximum(right - 1, 0)
    exact = source_wavelength[right] == wavelength
    left[exact] = right[exact]
    alpha = np.zeros(wavelength.shape)
    different = inside & (right != left)
    alpha[different] = (wavelength[different] - source_wavelength[left[different]]) / (
        source_wavelength[right[different]] - source_wavelength[left[different]]
    )
    flux = np.full(wavelength.shape, np.nan)
    uncertainty = np.full(wavelength.shape, np.nan)
    mask = np.full(wavelength.shape, TELLURIC_INVALID_BIT, dtype=np.uint16)
    flux[inside] = source_flux[left[inside]] + alpha[inside] * (
        source_flux[right[inside]] - source_flux[left[inside]]
    )
    variance = source_uncertainty**2
    variance_sum = variance[left] + variance[right]
    from_left = variance[left] + alpha**2 * variance_sum
    from_right = variance[right] + (1.0 - alpha) ** 2 * variance_sum
    uncertainty[inside] = np.sqrt(np.minimum(from_left[inside], from_right[inside]))
    mask[inside] = source_mask[left[inside]] | source_mask[right[inside]]
    mask[exact & inside] = source_mask[right[exact & inside]]
    return flux, uncertainty, mask


def apply_telluric_correction(
    science: CombinedSpectrum,
    correction: TelluricCorrection,
    *,
    science_group: str,
) -> TelluricCorrectedSpectrum:
    """Apply correction and propagate errors as ``xtellcor_finish.pro``."""

    correction_by_order = {order.order: order for order in correction.orders}
    corrected_orders: list[ExtractedOrder] = []
    for order in science.orders:
        if order.order not in correction_by_order:
            raise ValueError(f"telluric standard lacks science order {order.order}")
        multiplier, multiplier_uncertainty, multiplier_mask = _resample_correction(
            correction_by_order[order.order], order.wavelength_micron
        )
        corrected_flux = multiplier * order.flux
        corrected_uncertainty = np.sqrt(
            (multiplier * order.uncertainty) ** 2 + (order.flux * multiplier_uncertainty) ** 2
        )
        output_mask = order.mask | multiplier_mask
        invalid = ~np.isfinite(corrected_flux) | ~np.isfinite(corrected_uncertainty)
        output_mask[invalid] |= TELLURIC_INVALID_BIT
        corrected_orders.append(
            ExtractedOrder(
                order=order.order,
                wavelength_micron=order.wavelength_micron.copy(),
                flux=corrected_flux,
                uncertainty=corrected_uncertainty,
                mask=output_mask,
                trace_arcsec=np.full(order.wavelength_micron.shape, np.nan),
                background=np.full(order.wavelength_micron.shape, np.nan),
            )
        )
    return TelluricCorrectedSpectrum(
        orders=tuple(corrected_orders),
        science_group=science_group,
        standard_group=correction.standard_group,
        science_files=science.input_files,
        standard_files=correction.standard_files,
        science_airmass=None if science.metadata is None else science.metadata.mean_airmass,
        standard_airmass=correction.standard_airmass,
        science_metadata=science.metadata,
    )


def _order_hdu(order: ExtractedOrder, *, name: str, flux_unit: str) -> fits.BinTableHDU:
    return fits.BinTableHDU.from_columns(
        [
            fits.Column(name="WAVELENGTH", format="D", unit="um", array=order.wavelength_micron),
            fits.Column(name="FLUX", format="D", unit=flux_unit, array=order.flux),
            fits.Column(name="UNCERTAINTY", format="D", unit=flux_unit, array=order.uncertainty),
            fits.Column(name="MASK", format="I", array=order.mask),
        ],
        name=name,
    )


def write_telluric_correction(product: TelluricCorrection, path: str | Path) -> Path:
    """Write the multiplier and convolved Vega model for one standard group."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "TELLURIC"
    header["STDGRP"] = product.standard_group
    header["BMAG"] = product.b_magnitude
    header["VMAG"] = product.v_magnitude
    header["CATRV"] = product.catalog_rv_kms
    header["EARTHLSR"] = product.earth_lsr_velocity_kms
    header["VEGASHFT"] = product.applied_velocity_shift_kms
    header["AIRMASS"] = product.standard_airmass
    header["SLTW_ARC"] = product.slit_width_arcsec
    for index, value in enumerate(product.ip_coefficients):
        header[f"IPCOEF{index}"] = value
    header.add_history(f"VEGA_MODEL {product.vega_model_path}")
    header.add_history(f"IP_COEFFICIENTS {product.ip_coefficients_path}")
    for source in product.standard_files:
        header.add_history(f"STANDARD {source.name}")
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]
    for correction, vega in zip(product.orders, product.vega_orders, strict=True):
        hdus.append(
            _order_hdu(
                correction,
                name=f"ORDER{correction.order:03d}",
                flux_unit="erg cm-2 Angstrom-1 DN-1",
            )
        )
        hdus.append(
            _order_hdu(
                vega,
                name=f"VEGA{vega.order:03d}",
                flux_unit="erg s-1 cm-2 Angstrom-1",
            )
        )
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output


def write_corrected_spectrum(product: TelluricCorrectedSpectrum, path: str | Path) -> Path:
    """Write a checksummed telluric- and flux-corrected science spectrum."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "CORRECTED"
    header["SCIGRP"] = product.science_group
    header["STDGRP"] = product.standard_group
    if product.science_metadata is not None:
        header["OBSMODE"] = product.science_metadata.mode
        header["AVE_MJD"] = product.science_metadata.mean_mjd
        header["RA"] = product.science_metadata.ra
        header["DEC"] = product.science_metadata.dec
        header["SLTW_ARC"] = product.science_metadata.slit_width_arcsec
        header["PLTSCALE"] = product.science_metadata.plate_scale_arcsec_per_pixel
    if product.science_airmass is not None:
        header["AIRMASS"] = product.science_airmass
    if product.standard_airmass is not None:
        header["STDAMASS"] = product.standard_airmass
    if product.science_airmass is not None and product.standard_airmass is not None:
        header["AMDIFF"] = product.standard_airmass - product.science_airmass
    for source in product.science_files:
        header.add_history(f"SCIENCE {source.name}")
    for source in product.standard_files:
        header.add_history(f"STANDARD {source.name}")
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU(header=header)]
    hdus.extend(
        _order_hdu(
            order,
            name=f"ORDER{order.order:03d}",
            flux_unit="erg s-1 cm-2 Angstrom-1",
        )
        for order in product.orders
    )
    fits.HDUList(hdus).writeto(output, overwrite=True, checksum=True)
    return output
