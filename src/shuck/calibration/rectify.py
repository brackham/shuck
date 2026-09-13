"""Two-dimensional wavelength distortion and order rectification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from shuck.calibration.flat import NormalizedFlat, _robust_polynomial_fit
from shuck.calibration.wavecal import (
    WavecalInfo,
    WavelengthSolution,
    _fit_line,
    _robust_fit_polynomial_2d,
    evaluate_polynomial_2d,
)


@dataclass(frozen=True)
class ArcLineTrace:
    """Measured detector coordinates and straight-line model for one arc line."""

    order: int
    wavelength_micron: float
    x: np.ndarray
    y: np.ndarray
    coefficients: np.ndarray
    x_mid: float
    y_mid: float
    slope: float
    rms_pixels: float


@dataclass(frozen=True)
class RectificationGeometry:
    """Bilinear sampling coordinates for one rectified spectral order."""

    order: int
    reference_pixel: np.ndarray
    wavelength_micron: np.ndarray
    spatial_arcsec: np.ndarray
    x_index: np.ndarray
    y_index: np.ndarray


@dataclass(frozen=True)
class DistortionSolution:
    """Global line-slope model and per-order rectification coordinates."""

    traces: tuple[ArcLineTrace, ...]
    slope_coefficients: np.ndarray
    slope_fit_used: np.ndarray
    slope_x_degree: int
    slope_y_degree: int
    geometries: tuple[RectificationGeometry, ...]


@dataclass(frozen=True)
class RectifiedOrder:
    """One spectral order sampled on wavelength and slit-position axes."""

    order: int
    wavelength_micron: np.ndarray
    spatial_arcsec: np.ndarray
    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray


def _order_index(flat: NormalizedFlat, order: int) -> int:
    matches = np.flatnonzero(flat.orders == order)
    if len(matches) != 1:
        raise ValueError(f"order {order} is not uniquely present in normalized flat")
    return int(matches[0])


def _trace_direction(
    image: np.ndarray,
    x_limits: tuple[int, int],
    start_y: float,
    stop_y: float,
    direction: int,
    start_x: float,
    window_pixels: int,
    n_terms: int,
    y_step: int,
    y_sum: int,
    line_degree: int,
) -> tuple[list[float], list[float]]:
    measured_x: list[float] = []
    measured_y: list[float] = []
    guess = start_x
    y = start_y + direction * y_step
    while (y < stop_y - 1) if direction > 0 else (y > stop_y + 1):
        row = int(np.rint(y))
        low_row = max(0, row - y_sum)
        high_row = min(image.shape[0], row + y_sum + 1)
        half_window = window_pixels / 2
        low_x = max(x_limits[0], int(np.floor(guess - half_window)))
        high_x = min(x_limits[1], int(np.ceil(guess + half_window)))
        if high_x - low_x + 1 >= n_terms:
            pixels = np.arange(low_x, high_x + 1, dtype=np.float64)
            flux = np.nanmedian(image[low_row:high_row, low_x : high_x + 1], axis=0)
            center, _, _, found = _fit_line(pixels, flux, guess, window_pixels, n_terms)
            if found:
                measured_x.append(center)
                measured_y.append(y)
                if len(measured_x) > 3:
                    degree = min(line_degree, len(measured_x) - 1)
                    coefficients = _robust_polynomial_fit(
                        np.asarray(measured_y),
                        np.asarray(measured_x),
                        degree,
                        fractional_sigma_change=0.1,
                    )
                    guess = float(
                        np.polynomial.polynomial.polyval(y + direction * y_step, coefficients)
                    )
                else:
                    guess = center
        y += direction * y_step
    return measured_x, measured_y


def trace_arc_lines(
    solution: WavelengthSolution,
    flat: NormalizedFlat,
    info: WavecalInfo,
) -> tuple[ArcLineTrace, ...]:
    """Trace accepted ThAr lines across each slit.

    The behavioral references are SpeXTool 5.0.3 ``mc_findlines2d.pro``
    and ``mc_fitlines2dxd.pro``.
    """

    traces: list[ArcLineTrace] = []
    for line in solution.line_measurements:
        if not line.used:
            continue
        flat_index = _order_index(flat, line.order)
        start, stop = (int(value) for value in flat.xranges[flat_index])
        x0 = line.fitted_position
        bottom = float(np.polynomial.polynomial.polyval(x0, flat.edge_coefficients[flat_index, 0]))
        top = float(np.polynomial.polynomial.polyval(x0, flat.edge_coefficients[flat_index, 1]))
        middle = (bottom + top) / 2
        derivative = abs(
            float(solution.wavelength(line.order, np.array([x0 + 0.5]))[0])
            - float(solution.wavelength(line.order, np.array([x0 - 0.5]))[0])
        )
        window_pixels = max(line.n_terms, int(np.rint(line.window_angstrom / (derivative * 1e4))))

        up_x, up_y = _trace_direction(
            solution.arc_image,
            (start, stop),
            middle,
            top,
            1,
            x0,
            window_pixels,
            line.n_terms,
            info.find_y_step,
            info.find_y_sum,
            info.line_degree,
        )
        # SpeXTool rejects a line before tracing downward unless the upper
        # half supplies more than ten successful row measurements.
        if len(up_x) <= 10:
            continue
        preliminary = _robust_polynomial_fit(
            np.asarray(up_y),
            np.asarray(up_x),
            info.line_degree,
            fractional_sigma_change=0.1,
        )
        down_start = float(np.polynomial.polynomial.polyval(middle, preliminary))
        down_x, down_y = _trace_direction(
            solution.arc_image,
            (start, stop),
            middle,
            bottom,
            -1,
            down_start,
            window_pixels,
            line.n_terms,
            info.find_y_step,
            info.find_y_sum,
            info.line_degree,
        )
        trace_x = np.asarray(up_x + down_x)
        trace_y = np.asarray(up_y + down_y)
        if len(trace_x) < info.line_degree + 2:
            continue
        coefficients = _robust_polynomial_fit(
            trace_y,
            trace_x,
            info.line_degree,
            threshold=3.0,
            fractional_sigma_change=0.01,
        )
        model = np.polynomial.polynomial.polyval(trace_y, coefficients)
        rms = float(np.sqrt(np.mean((trace_x - model) ** 2)))
        traces.append(
            ArcLineTrace(
                order=line.order,
                wavelength_micron=line.wavelength_micron,
                x=trace_x,
                y=trace_y,
                coefficients=coefficients,
                x_mid=float(np.polynomial.polynomial.polyval(middle, coefficients)),
                y_mid=middle,
                slope=float(coefficients[1]),
                rms_pixels=rms,
            )
        )
    return tuple(traces)


def _edge_intersection(
    x_grid: np.ndarray,
    edge_coefficients: np.ndarray,
    line_coefficients: np.ndarray,
    reference_x: float,
) -> tuple[float, float]:
    edge_y = np.polynomial.polynomial.polyval(x_grid, edge_coefficients)
    difference = x_grid - np.polynomial.polynomial.polyval(edge_y, line_coefficients)
    crossing = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
    if len(crossing):
        candidates = []
        for index in crossing:
            denominator = difference[index + 1] - difference[index]
            fraction = 0.0 if denominator == 0 else -difference[index] / denominator
            candidates.append(x_grid[index] + fraction * (x_grid[index + 1] - x_grid[index]))
        closest = int(np.argmin(np.abs(np.asarray(candidates) - reference_x)))
        x_intersection = float(candidates[closest])
    else:
        nearest = int(np.argmin(np.abs(difference)))
        x_intersection = float(x_grid[nearest])
    y_intersection = float(np.polynomial.polynomial.polyval(x_intersection, edge_coefficients))
    return x_intersection, y_intersection


def make_rectification_geometry(
    flat: NormalizedFlat,
    wavelength_solution: WavelengthSolution,
    slope_coefficients: np.ndarray,
    info: WavecalInfo,
) -> tuple[RectificationGeometry, ...]:
    """Generate SpeXTool-equivalent detector sampling coordinates.

    The behavioral reference is SpeXTool 5.0.3 ``mc_mkrectindcs2d.pro``.
    """

    spatial = np.arange(
        0.0,
        flat.slit_height_arcsec + flat.plate_scale_arcsec_per_pixel / 2,
        flat.plate_scale_arcsec_per_pixel,
    )
    geometries: list[RectificationGeometry] = []
    for index, order in enumerate(flat.orders):
        start, stop = (int(value) for value in flat.xranges[index])
        reference = np.arange(start, stop + 1, dtype=np.float64)
        bottom = np.polynomial.polynomial.polyval(reference, flat.edge_coefficients[index, 0])
        top = np.polynomial.polynomial.polyval(reference, flat.edge_coefficients[index, 1])
        middle = (bottom + top) / 2
        x_index = np.empty((len(spatial), len(reference)), dtype=np.float64)
        y_index = np.empty_like(x_index)
        for column, reference_x in enumerate(reference):
            slope = float(
                evaluate_polynomial_2d(
                    np.array(reference_x),
                    np.array(middle[column]),
                    info.c1_x_degree,
                    info.c1_y_degree,
                    slope_coefficients,
                )
            )
            line_coefficients = np.array([reference_x - slope * middle[column], slope])
            _, y_bottom = _edge_intersection(
                reference, flat.edge_coefficients[index, 0], line_coefficients, reference_x
            )
            _, y_top = _edge_intersection(
                reference, flat.edge_coefficients[index, 1], line_coefficients, reference_x
            )
            fraction = spatial / flat.slit_height_arcsec
            current_y = y_bottom + fraction * (y_top - y_bottom)
            y_index[:, column] = current_y
            x_index[:, column] = np.polynomial.polynomial.polyval(current_y, line_coefficients)
        valid_columns = np.all(
            (x_index > start + 2) & (x_index < stop - 2),
            axis=0,
        )
        good = np.flatnonzero(valid_columns)
        if not len(good):
            raise ValueError(f"rectification trims every column from order {order}")
        selected = slice(good[0], good[-1] + 1)
        selected_reference = reference[selected]
        geometries.append(
            RectificationGeometry(
                order=int(order),
                reference_pixel=selected_reference,
                wavelength_micron=wavelength_solution.wavelength(int(order), selected_reference),
                spatial_arcsec=spatial.copy(),
                x_index=x_index[:, selected],
                y_index=y_index[:, selected],
            )
        )
    return tuple(geometries)


def build_distortion_solution(
    wavelength_solution: WavelengthSolution,
    flat: NormalizedFlat,
    info: WavecalInfo,
) -> DistortionSolution:
    """Fit the 2DXD distortion surface and construct rectification indices.

    Behavioral references are SpeXTool 5.0.3 ``mc_findlines2d.pro``,
    ``mc_fitlines2dxd.pro``, and ``mc_fitlinecoeffs2d.pro``.
    """

    if info.line_degree != 1:
        raise ValueError("v0.1 supports the straight-line iSHELL distortion model only")
    traces = trace_arc_lines(wavelength_solution, flat, info)
    required = (info.c1_x_degree + 1) * (info.c1_y_degree + 1)
    if len(traces) < required:
        raise ValueError(f"only {len(traces)} usable 2-D line traces; need at least {required}")
    x_mid = np.array([trace.x_mid for trace in traces])
    y_mid = np.array([trace.y_mid for trace in traces])
    slopes = np.array([trace.slope for trace in traces])
    coefficients, used = _robust_fit_polynomial_2d(
        x_mid,
        y_mid,
        slopes,
        info.c1_x_degree,
        info.c1_y_degree,
        threshold=3.5,
    )
    geometries = make_rectification_geometry(flat, wavelength_solution, coefficients, info)
    return DistortionSolution(
        traces=traces,
        slope_coefficients=coefficients,
        slope_fit_used=used,
        slope_x_degree=info.c1_x_degree,
        slope_y_degree=info.c1_y_degree,
        geometries=geometries,
    )


def rectify_order(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    geometry: RectificationGeometry,
) -> RectifiedOrder:
    """Bilinearly rectify an order following ``mc_rectifyorder.pro`` method 0."""

    coordinates = np.array([geometry.y_index, geometry.x_index])
    rectified_image = ndimage.map_coordinates(
        np.asarray(image, dtype=np.float64),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    rectified_variance = ndimage.map_coordinates(
        np.asarray(variance, dtype=np.float64),
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    # Preserve discrete bit patterns with nearest-neighbor sampling.
    rectified_mask = ndimage.map_coordinates(
        np.asarray(mask, dtype=np.uint16),
        coordinates,
        order=0,
        mode="constant",
        cval=np.iinfo(np.uint16).max,
        prefilter=False,
    ).astype(np.uint16)
    invalid = ~np.isfinite(rectified_image) | ~np.isfinite(rectified_variance)
    rectified_mask[invalid] |= np.uint16(1 << 15)
    return RectifiedOrder(
        order=geometry.order,
        wavelength_micron=geometry.wavelength_micron.copy(),
        spatial_arcsec=geometry.spatial_arcsec.copy(),
        image=rectified_image,
        variance=rectified_variance,
        mask=rectified_mask,
    )
