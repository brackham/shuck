"""ThAr wavelength calibration and rectification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import optimize, signal

from shuck.calibration.flat import (
    NormalizedFlat,
    _fill_invalid_nearest,
    rotate_to_processing,
)
from shuck.detector import IShellDetectorCalibration, process_raw_frame
from shuck.io import read_ishell_raw


@dataclass(frozen=True)
class WavecalInfo:
    """Mode-specific 2DXD metadata distributed with SpeXTool."""

    mode: str
    orders: np.ndarray
    reference_spectra: np.ndarray
    xranges: np.ndarray
    home_order: int
    dispersion_degree: int
    order_degree: int
    cross_correlation_order: int
    extraction_aperture_arcsec: float
    line_list_path: Path
    reference_coefficients: np.ndarray
    line_degree: int
    find_y_step: int
    find_y_sum: int
    generate_y_step: int
    c1_x_degree: int
    c1_y_degree: int
    source_path: Path


@dataclass(frozen=True)
class LineMeasurement:
    """One ThAr line-list entry and its measured detector position."""

    order: int
    wavelength_micron: float
    species: str
    window_angstrom: float
    fit_type: str
    n_terms: int
    guess_position: float
    fitted_position: float
    fwhm_pixels: float
    intensity: float
    found: bool
    used: bool
    residual_angstrom: float


@dataclass(frozen=True)
class WavelengthSolution:
    """A simultaneous cross-dispersed 1-D wavelength solution."""

    mode: str
    orders: np.ndarray
    xranges: np.ndarray
    coefficients: np.ndarray
    home_order: int
    dispersion_degree: int
    order_degree: int
    cross_correlation_offset: float
    rms_angstrom: float
    line_measurements: tuple[LineMeasurement, ...]
    arc_image: np.ndarray
    arc_variance: np.ndarray
    extracted_arc: tuple[np.ndarray, ...]
    input_files: tuple[Path, ...]
    rotation: int

    def wavelength(self, order: int, pixels: np.ndarray) -> np.ndarray:
        """Evaluate vacuum wavelength in microns for one order."""

        scaled = evaluate_polynomial_2d(
            np.asarray(pixels, dtype=np.float64),
            np.full_like(np.asarray(pixels, dtype=np.float64), order),
            self.dispersion_degree,
            self.order_degree,
            self.coefficients,
        )
        return scaled * self.home_order / order


def load_wavecal_info(spextool_directory: str | Path, mode: str) -> WavecalInfo:
    """Read SpeXTool's iSHELL ``*_wavecalinfo.fits`` reference product.

    The behavioral reference is SpeXTool 5.0.3 ``mc_readwavecalinfo.pro``.
    """

    if mode not in {"J3", "Kgas"}:
        raise ValueError(f"unsupported iSHELL v0.1 mode {mode!r}")
    data_directory = (
        Path(spextool_directory).expanduser().resolve() / "instruments" / "ishell" / "data"
    )
    path = data_directory / f"{mode}_wavecalinfo.fits"
    spectra, header = fits.getdata(path, header=True, memmap=False)
    if header["WCALTYPE"].strip() != "2DXD":
        raise ValueError(f"{path.name} is not a 2DXD wavelength reference")
    orders = np.array([int(value) for value in header["ORDERS"].split(",")], dtype=np.int16)
    xranges = np.array(
        [[int(value) for value in header[f"OR{order:03d}_XR"].split(",")] for order in orders],
        dtype=np.int32,
    )
    dispersion_degree = int(header["DISPDEG"])
    order_degree = int(header["ORDRDEG"])
    coefficients = np.array(
        [
            float(header[f"P2W_C{index:02d}"])
            for index in range((dispersion_degree + 1) * (order_degree + 1))
        ]
    )
    return WavecalInfo(
        mode=mode,
        orders=orders,
        reference_spectra=np.asarray(spectra, dtype=np.float64),
        xranges=xranges,
        home_order=int(header["HOMEORDR"]),
        dispersion_degree=dispersion_degree,
        order_degree=order_degree,
        cross_correlation_order=int(header["XCORORDR"]),
        extraction_aperture_arcsec=float(header["EXTAP"]),
        line_list_path=data_directory / header["LINELIST"].strip(),
        reference_coefficients=coefficients,
        line_degree=int(header["LINEDEG"]),
        find_y_step=int(header["FNDYSTEP"]),
        find_y_sum=int(header["FNDYSUM"]),
        generate_y_step=int(header["GENYSTEP"]),
        c1_x_degree=int(header["C1XDEG"]),
        c1_y_degree=int(header["C1YDEG"]),
        source_path=path,
    )


def read_line_list(path: str | Path) -> tuple[tuple[int, float, str, float, str, int], ...]:
    """Read the fields used from a SpeXTool pipe-delimited line list."""

    rows: list[tuple[int, float, str, float, str, int]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 6:
            raise ValueError(f"invalid SpeXTool line-list row: {raw_line}")
        rows.append(
            (
                int(fields[0]),
                float(fields[1]),
                fields[2],
                float(fields[3]),
                fields[4],
                int(fields[5]),
            )
        )
    return tuple(rows)


def evaluate_polynomial_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_degree: int,
    y_degree: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate the coefficient ordering used by SpeXTool ``mc_poly2d.pro``."""

    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.size != (x_degree + 1) * (y_degree + 1):
        raise ValueError("coefficient count does not match polynomial degrees")
    result = np.zeros_like(x_values)
    index = 0
    for y_exponent in range(y_degree + 1):
        for x_exponent in range(x_degree + 1):
            result += coeffs[index] * x_values**x_exponent * y_values**y_exponent
            index += 1
    return result


def _extract_arc_spectra(
    image: np.ndarray,
    variance: np.ndarray,
    flat: NormalizedFlat,
    wavecal_info: WavecalInfo,
) -> tuple[np.ndarray, ...]:
    spectra: list[np.ndarray] = []
    half_aperture = wavecal_info.extraction_aperture_arcsec / 2
    for order in wavecal_info.orders:
        flat_index = int(np.flatnonzero(flat.orders == order)[0])
        start, stop = flat.xranges[flat_index]
        x = np.arange(start, stop + 1)
        bottom = np.polynomial.polynomial.polyval(x, flat.edge_coefficients[flat_index, 0])
        top = np.polynomial.polynomial.polyval(x, flat.edge_coefficients[flat_index, 1])
        flux = np.full(len(x), np.nan)
        uncertainty = np.full(len(x), np.nan)
        for index, column in enumerate(x):
            center = (bottom[index] + top[index]) / 2
            radius_pixels = half_aperture * (top[index] - bottom[index]) / flat.slit_height_arcsec
            low = max(0, int(np.ceil(center - radius_pixels)))
            high = min(image.shape[0], int(np.floor(center + radius_pixels)) + 1)
            good = (
                np.isfinite(image[low:high, column])
                & np.isfinite(variance[low:high, column])
                & (variance[low:high, column] >= 0)
            )
            if np.any(good):
                flux[index] = np.sum(image[low:high, column][good])
                uncertainty[index] = np.sqrt(np.sum(variance[low:high, column][good]))
        spectra.append(np.column_stack((x, flux, uncertainty)))
    return tuple(spectra)


def cross_correlation_offset(
    anchor_x: np.ndarray,
    anchor_flux: np.ndarray,
    observed_x: np.ndarray,
    observed_flux: np.ndarray,
) -> float:
    """Determine the automatic pixel offset used by ``xmc_corspec.pro``."""

    x_anchor = np.asarray(anchor_x, dtype=np.float64)
    reference = np.asarray(anchor_flux, dtype=np.float64)
    observed = np.interp(x_anchor, observed_x, observed_flux, left=0.0, right=0.0)
    reference = np.where(np.isfinite(reference), reference, 0.0)
    observed = np.where(np.isfinite(observed), observed, 0.0)
    reference -= np.mean(reference)
    observed -= np.mean(observed)
    correlation = signal.correlate(observed, reference, mode="full", method="fft")
    lags = signal.correlation_lags(len(observed), len(reference), mode="full")
    peak = int(np.argmax(correlation))
    window = slice(max(0, peak - 4), min(len(correlation), peak + 5))
    fit_lags = lags[window].astype(np.float64)
    fit_values = correlation[window]

    def lorentzian(
        x: np.ndarray, amplitude: float, center: float, width: float, base: float
    ) -> np.ndarray:
        return base + amplitude / (1.0 + ((x - center) / width) ** 2)

    try:
        parameters, _ = optimize.curve_fit(
            lorentzian,
            fit_lags,
            fit_values,
            p0=(fit_values.max() - np.median(fit_values), lags[peak], 1.0, np.median(fit_values)),
            bounds=([0, lags[peak] - 3, 0.05, -np.inf], [np.inf, lags[peak] + 3, 20, np.inf]),
            maxfev=10_000,
        )
        return float(parameters[1])
    except (RuntimeError, ValueError):
        return float(lags[peak])


def _gaussian_model(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    constant: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + constant + slope * (x - center)


def _fit_line(
    x: np.ndarray,
    flux: np.ndarray,
    guess: float,
    window_pixels: int,
    n_terms: int,
) -> tuple[float, float, float, bool]:
    selected = np.abs(x - guess) <= window_pixels / 2
    selected &= np.isfinite(flux)
    if np.count_nonzero(selected) < n_terms:
        return np.nan, np.nan, np.nan, False
    xx = x[selected]
    yy = flux[selected]
    baseline = np.median((yy[0], yy[-1]))
    amplitude = float(np.max(yy) - baseline)
    center = float(xx[np.argmax(yy)])
    if amplitude <= 0:
        return np.nan, np.nan, np.nan, False
    try:
        parameters, _ = optimize.curve_fit(
            _gaussian_model,
            xx,
            yy,
            p0=(amplitude, center, 1.5, baseline if n_terms >= 4 else 0.0, 0.0),
            bounds=(
                [0.0, guess - window_pixels / 2, 0.1, -np.inf, -np.inf],
                [np.inf, guess + window_pixels / 2, max(1.0, window_pixels), np.inf, np.inf],
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, False
    fitted_center = float(parameters[1])
    fwhm = float(2.354820045 * abs(parameters[2]))
    fitted_amplitude = float(parameters[0])
    found = abs(fitted_center - guess) <= 2 and fwhm > 0 and fitted_amplitude > 0
    return fitted_center, fwhm, fitted_amplitude, found


def _polynomial_design(x: np.ndarray, y: np.ndarray, x_degree: int, y_degree: int) -> np.ndarray:
    return np.column_stack(
        [
            x**x_exponent * y**y_exponent
            for y_exponent in range(y_degree + 1)
            for x_exponent in range(x_degree + 1)
        ]
    )


def _robust_fit_polynomial_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_degree: int,
    y_degree: int,
    *,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a clipped 2-D polynomial without losing precision to raw coordinates."""

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(good) < (x_degree + 1) * (y_degree + 1):
        raise ValueError("too few finite samples for 2-D polynomial fit")
    x_center = float(np.mean(x[good]))
    x_scale = float(np.std(x[good]))
    y_center = float(np.mean(y[good]))
    y_scale = float(np.std(y[good]))
    if x_scale == 0 or y_scale == 0:
        raise ValueError("2-D polynomial coordinates must span both dimensions")
    normalized_x = (x - x_center) / x_scale
    normalized_y = (y - y_center) / y_scale
    design = _polynomial_design(normalized_x, normalized_y, x_degree, y_degree)
    for _ in range(10):
        normalized_coefficients, *_ = np.linalg.lstsq(design[good], z[good], rcond=None)
        residual = z - design @ normalized_coefficients
        median = np.median(residual[good])
        mad = 1.482 * np.median(np.abs(residual[good] - median))
        if mad == 0:
            break
        updated = good & (np.abs((residual - median) / mad) <= threshold)
        if np.count_nonzero(updated) < design.shape[1]:
            break
        if np.array_equal(updated, good):
            break
        good = updated
    normalized_coefficients, *_ = np.linalg.lstsq(design[good], z[good], rcond=None)
    normalized_coefficients = normalized_coefficients.reshape(y_degree + 1, x_degree + 1)
    raw_coefficients = np.zeros_like(normalized_coefficients)
    for y_exponent in range(y_degree + 1):
        for x_exponent in range(x_degree + 1):
            coefficient = normalized_coefficients[y_exponent, x_exponent]
            for raw_y_exponent in range(y_exponent + 1):
                y_factor = (
                    comb(y_exponent, raw_y_exponent)
                    * (-y_center) ** (y_exponent - raw_y_exponent)
                    / y_scale**y_exponent
                )
                for raw_x_exponent in range(x_exponent + 1):
                    x_factor = (
                        comb(x_exponent, raw_x_exponent)
                        * (-x_center) ** (x_exponent - raw_x_exponent)
                        / x_scale**x_exponent
                    )
                    raw_coefficients[raw_y_exponent, raw_x_exponent] += (
                        coefficient * y_factor * x_factor
                    )
    return raw_coefficients.ravel(), good


def _robust_fit_wavelength(
    x: np.ndarray,
    order: np.ndarray,
    scaled_wavelength: np.ndarray,
    x_degree: int,
    order_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _robust_fit_polynomial_2d(
        x,
        order,
        scaled_wavelength,
        x_degree,
        order_degree,
        threshold=3.0,
    )


def build_wavelength_solution(
    arc_on_files: Sequence[str | Path],
    arc_off_files: Sequence[str | Path],
    detector_calibration: IShellDetectorCalibration,
    normalized_flat: NormalizedFlat,
    wavecal_info: WavecalInfo,
) -> WavelengthSolution:
    """Build the supported automatic ThAr 1DXD wavelength solution.

    Behavioral references are the arc path in ``mc_ishellcals2dxd.pro``,
    ``xmc_corspec.pro``, ``mc_getlinexguess.pro``,
    ``mc_findlines1dxd.pro``, and ``mc_wavecal1dxd.pro``.
    """

    on_paths = tuple(Path(path) for path in arc_on_files)
    off_paths = tuple(Path(path) for path in arc_off_files)
    if not on_paths or len(on_paths) != len(off_paths):
        raise ValueError("ThAr on/off inputs must be non-empty one-to-one pairs")
    pair_images: list[np.ndarray] = []
    pair_variances: list[np.ndarray] = []
    for on_path, off_path in zip(on_paths, off_paths, strict=True):
        on = process_raw_frame(read_ishell_raw(on_path), detector_calibration)
        off = process_raw_frame(read_ishell_raw(off_path), detector_calibration)
        if on.metadata.mode != wavecal_info.mode or off.metadata.mode != wavecal_info.mode:
            raise ValueError("arc mode does not match wavelength-calibration mode")
        with np.errstate(invalid="ignore"):
            difference = on.image - off.image
        pair_images.append(rotate_to_processing(difference, normalized_flat.rotation))
        pair_variances.append(
            rotate_to_processing(on.variance + off.variance, normalized_flat.rotation)
        )
    arc = np.mean(pair_images, axis=0) / normalized_flat.image
    arc_variance = np.sum(pair_variances, axis=0) / len(pair_variances) ** 2
    arc_variance /= normalized_flat.image**2
    invalid = (normalized_flat.mask != 0) | ~np.isfinite(arc)
    arc[invalid] = np.nan
    arc_variance[invalid] = np.nan
    repaired_arc = _fill_invalid_nearest(arc)
    extracted = _extract_arc_spectra(repaired_arc, arc_variance, normalized_flat, wavecal_info)

    cross_index = int(
        np.flatnonzero(wavecal_info.orders == wavecal_info.cross_correlation_order)[0]
    )
    reference = wavecal_info.reference_spectra[cross_index]
    reference_pixels = np.arange(reference.shape[1]) + wavecal_info.xranges[cross_index, 0]
    offset = cross_correlation_offset(
        reference_pixels,
        reference[1],
        extracted[cross_index][:, 0],
        extracted[cross_index][:, 1],
    )

    preliminary: list[dict[str, float | int | str | bool]] = []
    for order, wavelength, species, window, fit_type, n_terms in read_line_list(
        wavecal_info.line_list_path
    ):
        order_index = int(np.flatnonzero(wavecal_info.orders == order)[0])
        reference_wavelength = wavecal_info.reference_spectra[order_index, 0]
        reference_pixels = (
            np.arange(len(reference_wavelength)) + wavecal_info.xranges[order_index, 0]
        )
        finite = np.isfinite(reference_wavelength)
        guess = float(
            np.interp(wavelength, reference_wavelength[finite], reference_pixels[finite]) + offset
        )
        left = np.interp(
            wavelength - window / 2e4, reference_wavelength[finite], reference_pixels[finite]
        )
        right = np.interp(
            wavelength + window / 2e4, reference_wavelength[finite], reference_pixels[finite]
        )
        window_pixels = max(n_terms, int(np.rint(abs(right - left))))
        spectrum = extracted[order_index]
        position, fwhm, intensity, found = _fit_line(
            spectrum[:, 0], spectrum[:, 1], guess, window_pixels, n_terms
        )
        preliminary.append(
            {
                "order": order,
                "wavelength": wavelength,
                "species": species,
                "window": window,
                "fit_type": fit_type,
                "n_terms": n_terms,
                "guess": guess,
                "position": position,
                "fwhm": fwhm,
                "intensity": intensity,
                "found": found,
            }
        )

    found = np.array([bool(line["found"]) for line in preliminary])
    positions = np.array([float(line["position"]) for line in preliminary])
    orders = np.array([int(line["order"]) for line in preliminary])
    wavelengths = np.array([float(line["wavelength"]) for line in preliminary])
    coefficients, fit_good = _robust_fit_wavelength(
        positions[found],
        orders[found],
        wavelengths[found] * orders[found] / wavecal_info.home_order,
        wavecal_info.dispersion_degree,
        wavecal_info.order_degree,
    )
    used = np.zeros(len(preliminary), dtype=bool)
    used[np.flatnonzero(found)] = fit_good
    scaled_model = evaluate_polynomial_2d(
        positions,
        orders,
        wavecal_info.dispersion_degree,
        wavecal_info.order_degree,
        coefficients,
    )
    residuals = (wavelengths * orders / wavecal_info.home_order - scaled_model) * 1e4
    rms = float(np.std(residuals[used], ddof=1))
    measurements = tuple(
        LineMeasurement(
            order=int(line["order"]),
            wavelength_micron=float(line["wavelength"]),
            species=str(line["species"]),
            window_angstrom=float(line["window"]),
            fit_type=str(line["fit_type"]),
            n_terms=int(line["n_terms"]),
            guess_position=float(line["guess"]),
            fitted_position=float(line["position"]),
            fwhm_pixels=float(line["fwhm"]),
            intensity=float(line["intensity"]),
            found=bool(line["found"]),
            used=bool(used[index]),
            residual_angstrom=float(residuals[index]),
        )
        for index, line in enumerate(preliminary)
    )
    return WavelengthSolution(
        mode=wavecal_info.mode,
        orders=wavecal_info.orders.copy(),
        xranges=normalized_flat.xranges.copy(),
        coefficients=coefficients,
        home_order=wavecal_info.home_order,
        dispersion_degree=wavecal_info.dispersion_degree,
        order_degree=wavecal_info.order_degree,
        cross_correlation_offset=offset,
        rms_angstrom=rms,
        line_measurements=measurements,
        arc_image=arc,
        arc_variance=arc_variance,
        extracted_arc=extracted,
        input_files=on_paths + off_paths,
        rotation=normalized_flat.rotation,
    )


def write_wavelength_solution(
    product: WavelengthSolution,
    path: str | Path,
    *,
    calib_id: str,
) -> Path:
    """Write a 1DXD wavelength solution, line fits, and arc data to FITS."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "WAVECAL_1DXD"
    header["CALIBID"] = calib_id
    header["MODE"] = product.mode
    header["ROTATION"] = product.rotation
    header["HOMEORD"] = product.home_order
    header["DISPDEG"] = product.dispersion_degree
    header["ORDRDEG"] = product.order_degree
    header["XCOROFF"] = product.cross_correlation_offset
    header["FITRMS"] = product.rms_angstrom
    header["NFOUND"] = sum(line.found for line in product.line_measurements)
    header["NUSED"] = sum(line.used for line in product.line_measurements)
    for index, coefficient in enumerate(product.coefficients):
        header[f"P2W_C{index:02d}"] = coefficient
    for source in product.input_files:
        header.add_history(f"INPUT {source.name}")

    maximum_pixels = int(np.max(product.xranges[:, 1] - product.xranges[:, 0] + 1))
    wavelength = np.full((len(product.orders), maximum_pixels), np.nan)
    pixel = np.full_like(wavelength, np.nan)
    for index, order in enumerate(product.orders):
        current_pixels = np.arange(product.xranges[index, 0], product.xranges[index, 1] + 1)
        pixel[index, : len(current_pixels)] = current_pixels
        wavelength[index, : len(current_pixels)] = product.wavelength(int(order), current_pixels)

    lines = product.line_measurements
    line_columns = [
        fits.Column(name="ORDER", format="I", array=[line.order for line in lines]),
        fits.Column(
            name="WAVELENGTH_UM",
            format="D",
            array=[line.wavelength_micron for line in lines],
        ),
        fits.Column(name="SPECIES", format="16A", array=[line.species for line in lines]),
        fits.Column(name="GUESS_X", format="D", array=[line.guess_position for line in lines]),
        fits.Column(name="FITTED_X", format="D", array=[line.fitted_position for line in lines]),
        fits.Column(name="FWHM_PIX", format="D", array=[line.fwhm_pixels for line in lines]),
        fits.Column(name="INTENSITY", format="D", array=[line.intensity for line in lines]),
        fits.Column(name="FOUND", format="L", array=[line.found for line in lines]),
        fits.Column(name="USED", format="L", array=[line.used for line in lines]),
        fits.Column(
            name="RESIDUAL_A",
            format="D",
            array=[line.residual_angstrom for line in lines],
        ),
    ]
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(np.fliplr(product.arc_image).astype(np.float32), header),
            fits.ImageHDU(np.fliplr(product.arc_variance).astype(np.float32), name="VARIANCE"),
            fits.ImageHDU(wavelength, name="WAVELENGTH"),
            fits.ImageHDU(pixel, name="PIXEL"),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="ORDER", format="I", array=product.orders),
                    fits.Column(name="XRANGE", format="2J", array=product.xranges),
                ],
                name="ORDERS",
            ),
            fits.BinTableHDU.from_columns(line_columns, name="LINES"),
        ]
    )
    hdus[2].header["BUNIT"] = "um"
    hdus.writeto(output, overwrite=True, checksum=True)
    return output
