"""Normalized-flat construction."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

from shuck.detector import IShellDetectorCalibration, process_raw_frame
from shuck.io import read_ishell_raw


@dataclass(frozen=True)
class FlatInfo:
    """Mode-specific flat/order metadata distributed with SpeXTool."""

    mode: str
    rotation: int
    slit_height_arcsec: float
    slit_height_pixels: float
    slit_height_range: tuple[int, int]
    orders: np.ndarray
    plate_scale_arcsec_per_pixel: float
    resolving_power_pixels: float
    step: int
    edge_fraction: float
    com_window: int
    edge_degree: int
    norm_nxgrid: int
    norm_nygrid: int
    oversample: float
    ybuffer: int
    ycor_order: int
    xranges: np.ndarray
    edge_coefficients: np.ndarray
    order_mask_native: np.ndarray
    source_path: Path


@dataclass(frozen=True)
class NormalizedFlat:
    """A normalized flat plus the measured geometry of its orders."""

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    combined_image: np.ndarray
    model: np.ndarray
    orders: np.ndarray
    xranges: np.ndarray
    edge_coefficients: np.ndarray
    order_rms: np.ndarray
    vertical_offset: int
    input_files: tuple[Path, ...]
    mode: str
    rotation: int
    plate_scale_arcsec_per_pixel: float
    slit_height_arcsec: float
    image_unit: str = "dimensionless"


def rotate_to_processing(image: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the iSHELL v0.1 IDL rotation in NumPy coordinates.

    Both supported modes use IDL ``ROTATE(..., 5)``. Comparing SpeXTool's
    stored order masks to its edge polynomials establishes that this is a
    left-right flip after FITS arrays are represented as NumPy arrays.
    """

    if rotation != 5:
        raise ValueError(f"v0.1 only supports the iSHELL ROTATION=5 path, not {rotation}")
    return np.fliplr(np.asarray(image))


def unrotate_from_processing(image: np.ndarray, rotation: int) -> np.ndarray:
    """Undo :func:`rotate_to_processing` for the supported iSHELL modes."""

    return rotate_to_processing(image, rotation)


def load_flat_info(spextool_directory: str | Path, mode: str) -> FlatInfo:
    """Read a SpeXTool iSHELL ``*_flatinfo.fits`` calibration.

    The behavioral reference is SpeXTool 5.0.3 ``mc_readflatinfo.pro``.
    """

    canonical_mode = mode.strip()
    if canonical_mode not in {"J3", "Kgas"}:
        raise ValueError(f"unsupported iSHELL v0.1 mode {mode!r}")
    path = (
        Path(spextool_directory).expanduser().resolve()
        / "instruments"
        / "ishell"
        / "data"
        / f"{canonical_mode}_flatinfo.fits"
    )
    order_mask, header = fits.getdata(path, header=True, memmap=False)
    orders = np.array([int(value) for value in header["ORDERS"].split(",")], dtype=np.int16)
    degree = int(header["EDGEDEG"])
    xranges = np.empty((len(orders), 2), dtype=np.int32)
    coefficients = np.empty((len(orders), 2, degree + 1), dtype=np.float64)
    for index, order in enumerate(orders):
        prefix = f"OR{order:03d}"
        xranges[index] = [int(value) for value in header[f"{prefix}_XR"].split(",")]
        for edge_index, edge_name in enumerate(("B", "T")):
            coefficients[index, edge_index] = [
                float(header[f"{prefix}_{edge_name}{coefficient + 1}"])
                for coefficient in range(degree + 1)
            ]

    slit_range = tuple(int(value) for value in header["SLTH_RNG"].split(","))
    return FlatInfo(
        mode=canonical_mode,
        rotation=int(header["ROTATION"]),
        slit_height_arcsec=float(header["SLTH_ARC"]),
        slit_height_pixels=float(header["SLTH_PIX"]),
        slit_height_range=(slit_range[0], slit_range[1]),
        orders=orders,
        plate_scale_arcsec_per_pixel=float(header["PLTSCALE"]),
        resolving_power_pixels=float(header["RPPIX"]),
        step=int(header["STEP"]),
        edge_fraction=float(header["FLATFRAC"]),
        com_window=int(header["COMWIN"]),
        edge_degree=degree,
        norm_nxgrid=int(header["NORM_NXG"]),
        norm_nygrid=int(header["NORM_NYG"]),
        oversample=float(header["OVERSAMP"]),
        ybuffer=int(header["YBUFFER"]),
        ycor_order=int(header["YCORORDR"]),
        xranges=xranges,
        edge_coefficients=coefficients,
        order_mask_native=np.asarray(order_mask, dtype=np.int16),
        source_path=path,
    )


def find_vertical_order_offset(flat: np.ndarray, info: FlatInfo) -> int:
    """Cross-correlate the designated order against SpeXTool's order mask.

    This is the automatic path in SpeXTool 5.0.3 ``mc_adjustguesspos.pro``.
    The returned sign is the value subtracted from reference edge positions.
    """

    image = np.asarray(flat, dtype=np.float64)
    image = np.where(np.isfinite(image), image, 0.0)
    order_mask = rotate_to_processing(info.order_mask_native, info.rotation) == info.ycor_order
    order_index = int(np.flatnonzero(info.orders == info.ycor_order)[0])
    start, stop = info.xranges[order_index]
    x = np.arange(start, stop + 1)
    bottom = np.polynomial.polynomial.polyval(x, info.edge_coefficients[order_index, 0])
    top = np.polynomial.polynomial.polyval(x, info.edge_coefficients[order_index, 1])
    slit_height = int(np.ceil(np.max(top - bottom)))
    lower = max(0, int(np.rint(np.min(bottom) - slit_height)))
    upper = min(image.shape[0] - 1, int(np.rint(np.max(top) + slit_height)))
    subflat = image[lower : upper + 1]
    submask = order_mask[lower : upper + 1]
    n_shifts = int(slit_height * 1.8 + 1)
    shifts = np.arange(n_shifts, dtype=int) - int(n_shifts / 2)
    scores = np.empty(len(shifts), dtype=np.float64)
    for index, shift in enumerate(shifts):
        data_start = max(-shift, 0)
        data_stop = min(subflat.shape[0] - shift, subflat.shape[0])
        mask_start = max(shift, 0)
        mask_stop = min(submask.shape[0] + shift, submask.shape[0])
        scores[index] = np.nansum(subflat[data_start:data_stop] * submask[mask_start:mask_stop])
    return int(shifts[int(np.argmax(scores))])


def _robust_polynomial_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    *,
    threshold: float = 3.0,
    fractional_sigma_change: float = 0.01,
) -> np.ndarray:
    """Port the iterative clipping used by ``mc_robustpoly1d.pro``."""

    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < degree + 1:
        raise ValueError("too few finite edge samples for polynomial fit")
    previous_sigma = np.inf
    for _ in range(10):
        fit = np.polynomial.polynomial.polyfit(x[good], y[good], degree)
        residual = y - np.polynomial.polynomial.polyval(x, fit)
        center = np.mean(residual[good])
        sigma = np.std(residual[good], ddof=1)
        if not np.isfinite(sigma) or sigma == 0:
            break
        updated = np.isfinite(residual) & (np.abs((residual - center) / sigma) <= threshold)
        if np.array_equal(updated, good):
            break
        if (
            np.isfinite(previous_sigma)
            and (previous_sigma - sigma) / previous_sigma < fractional_sigma_change
        ):
            break
        good = updated
        previous_sigma = sigma
    return np.polynomial.polynomial.polyfit(x[good], y[good], degree)


def _fill_invalid_nearest(image: np.ndarray) -> np.ndarray:
    """Fill invalid samples from the nearest finite pixel for interpolation."""

    source = np.asarray(image, dtype=np.float64)
    invalid = ~np.isfinite(source)
    if not np.any(invalid):
        return source.copy()
    if np.all(invalid):
        raise ValueError("cannot interpolate an image with no finite pixels")
    nearest = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return source[tuple(nearest)]


def find_order_edges(
    flat: np.ndarray,
    info: FlatInfo,
    vertical_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate and robustly fit order edges as ``mc_findorders.pro`` does."""

    image = np.asarray(flat, dtype=np.float64)
    finite_image = _fill_invalid_nearest(image)
    scaled = finite_image * (1000.0 / np.max(finite_image))
    edge_image = np.hypot(ndimage.sobel(scaled, axis=0), ndimage.sobel(scaled, axis=1))
    coefficients = np.empty_like(info.edge_coefficients)
    xranges = np.empty_like(info.xranges)
    rows = np.arange(image.shape[0], dtype=np.float64)
    half_window = int(np.rint(info.com_window / 2.0))

    for order_index in range(len(info.orders)):
        search_start, search_stop = info.xranges[order_index]
        starts = search_start + info.step - 1
        stops = search_stop - info.step + 1
        sample_columns = np.arange(starts, stops + 1, info.step, dtype=int)
        edges = np.full((len(sample_columns), 2), np.nan)
        centers = np.full(len(sample_columns), np.nan)
        guess_x = int(np.rint(np.mean(info.xranges[order_index])))
        reference_edges = [
            np.polynomial.polynomial.polyval(
                guess_x,
                info.edge_coefficients[order_index, edge],
            )
            - vertical_offset
            for edge in range(2)
        ]
        guess_y = float(np.mean(reference_edges))
        guess_index = int(np.argmin(np.abs(sample_columns - guess_x)))
        low = max(0, guess_index - info.edge_degree)
        high = min(len(centers), guess_index + info.edge_degree + 1)
        centers[low:high] = guess_y

        for indices in (
            range(guess_index, -1, -1),
            range(guess_index + 1, len(sample_columns)),
        ):
            top_active = bottom_active = True
            for sample_index in indices:
                column = sample_columns[sample_index]
                finite = np.isfinite(centers)
                center_degree = max(1, info.edge_degree - 2)
                if np.count_nonzero(finite) >= center_degree + 1:
                    center_fit = np.polynomial.polynomial.polyfit(
                        sample_columns[finite], centers[finite], center_degree
                    )
                    y_guess = np.polynomial.polynomial.polyval(column, center_fit)
                else:
                    y_guess = guess_y
                y_guess = int(
                    np.clip(np.rint(y_guess), info.ybuffer, image.shape[0] - info.ybuffer - 1)
                )
                flux_column = image[:, column]
                edge_column = edge_image[:, column]
                guess_flux = flux_column[y_guess]

                found = [np.nan, np.nan]
                below = np.flatnonzero(
                    (flux_column < info.edge_fraction * guess_flux) & (rows < y_guess)
                )
                above = np.flatnonzero(
                    (flux_column < info.edge_fraction * guess_flux) & (rows > y_guess)
                )
                candidates = (
                    below[-1] if len(below) and bottom_active else None,
                    above[0] if len(above) and top_active else None,
                )
                for edge_index, candidate in enumerate(candidates):
                    if candidate is None:
                        continue
                    window_start = max(0, candidate - half_window)
                    window_stop = min(image.shape[0], candidate + half_window + 1)
                    weights = edge_column[window_start:window_stop]
                    denominator = np.sum(weights)
                    if denominator > 0:
                        found[edge_index] = (
                            np.sum(rows[window_start:window_stop] * weights) / denominator
                        )

                if np.all(np.isfinite(found)):
                    height = found[1] - found[0]
                    if info.slit_height_range[0] < height < info.slit_height_range[1]:
                        edges[sample_index] = found
                        centers[sample_index] = np.mean(found)
                else:
                    centers[sample_index] = y_guess
                if np.isfinite(found[1]) and (
                    found[1] <= info.ybuffer or found[1] >= image.shape[0] - info.ybuffer - 1
                ):
                    top_active = False
                if np.isfinite(found[0]) and (
                    found[0] <= info.ybuffer or found[0] >= image.shape[0] - info.ybuffer - 1
                ):
                    bottom_active = False

        for edge_index in range(2):
            coefficients[order_index, edge_index] = _robust_polynomial_fit(
                sample_columns,
                edges[:, edge_index],
                info.edge_degree,
            )
        all_columns = np.arange(search_start, search_stop + 1)
        bottom = np.polynomial.polynomial.polyval(all_columns, coefficients[order_index, 0])
        top = np.polynomial.polynomial.polyval(all_columns, coefficients[order_index, 1])
        on_detector = (bottom > 0) & (top < image.shape[0] - 1)
        if not np.any(on_detector):
            raise ValueError(f"measured order {info.orders[order_index]} does not lie on detector")
        xranges[order_index] = [all_columns[on_detector][0], all_columns[on_detector][-1]]
    return coefficients, xranges


def _smooth_fiterpolate(image: np.ndarray, nx_cells: int, ny_cells: int) -> np.ndarray:
    """Port the quadratic-grid/bicubic surface in ``mc_fiterpolate.pro``."""

    source = np.asarray(image, dtype=np.float64)
    nx_points = nx_cells + 1
    ny_points = ny_cells + 1
    gx = np.rint(np.linspace(0, source.shape[1] - 1, nx_points)).astype(int)
    gy = np.rint(np.linspace(0, source.shape[0] - 1, ny_points)).astype(int)
    values = np.empty((ny_points, nx_points, 4), dtype=np.float64)
    for ix in range(nx_points):
        x1 = int((gx[max(ix - 1, 0)] + gx[ix]) / 2)
        x2 = int((gx[ix] + gx[min(nx_points - 1, ix + 1)]) / 2)
        for iy in range(ny_points):
            y1 = int((gy[max(iy - 1, 0)] + gy[iy]) / 2)
            y2 = int((gy[iy] + gy[min(ny_points - 1, iy + 1)]) / 2)
            patch = source[y1 : y2 + 1, x1 : x2 + 1]
            yy, xx = np.indices(patch.shape, dtype=np.float64)
            finite = np.isfinite(patch)
            design = np.column_stack(
                (
                    np.ones(np.count_nonzero(finite)),
                    xx[finite],
                    yy[finite],
                    xx[finite] ** 2,
                    yy[finite] ** 2,
                    xx[finite] * yy[finite],
                )
            )
            fit, *_ = np.linalg.lstsq(design, patch[finite], rcond=None)
            x = gx[ix] - x1
            y = gy[iy] - y1
            values[iy, ix] = (
                fit[0] + fit[1] * x + fit[2] * y + fit[3] * x**2 + fit[4] * y**2 + fit[5] * x * y,
                fit[1] + 2 * fit[3] * x + fit[5] * y,
                fit[2] + 2 * fit[4] * y + fit[5] * x,
                fit[5],
            )

    result = np.empty_like(source)
    hermite = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]],
        dtype=np.float64,
    )
    for ix in range(nx_cells):
        for iy in range(ny_cells):
            dx = gx[ix + 1] - gx[ix]
            dy = gy[iy + 1] - gy[iy]
            f00, fx00, fy00, fxy00 = values[iy, ix]
            f10, fx10, fy10, fxy10 = values[iy, ix + 1]
            f01, fx01, fy01, fxy01 = values[iy + 1, ix]
            f11, fx11, fy11, fxy11 = values[iy + 1, ix + 1]
            p = np.array(
                [
                    [f00, f01, fy00 * dy, fy01 * dy],
                    [f10, f11, fy10 * dy, fy11 * dy],
                    [fx00 * dx, fx01 * dx, fxy00 * dx * dy, fxy01 * dx * dy],
                    [fx10 * dx, fx11 * dx, fxy10 * dx * dy, fxy11 * dx * dy],
                ]
            )
            cell_coefficients = hermite @ p @ hermite.T
            x_values = np.arange(gx[ix], gx[ix + 1] + 1)
            y_values = np.arange(gy[iy], gy[iy + 1] + 1)
            tx = (x_values - gx[ix]) / dx
            ty = (y_values - gy[iy]) / dy
            x_basis = np.stack((np.ones_like(tx), tx, tx**2, tx**3), axis=1)
            y_basis = np.stack((np.ones_like(ty), ty, ty**2, ty**3), axis=1)
            result[np.ix_(y_values, x_values)] = y_basis @ cell_coefficients.T @ x_basis.T
    return result


def normalize_spectral_flat(
    combined_flat: np.ndarray,
    combined_variance: np.ndarray,
    edge_coefficients: np.ndarray,
    xranges: np.ndarray,
    info: FlatInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize orders following SpeXTool ``mc_normspecflat.pro``."""

    flat = np.asarray(combined_flat, dtype=np.float64)
    variance = np.asarray(combined_variance, dtype=np.float64)
    normalized = np.ones_like(flat)
    output_variance = np.full_like(variance, np.nan)
    model_image = np.ones_like(flat)
    order_rms = np.empty(len(info.orders), dtype=np.float64)

    for order_index in range(len(info.orders)):
        start, stop = xranges[order_index]
        x = np.arange(start, stop + 1)
        bottom = np.polynomial.polynomial.polyval(x, edge_coefficients[order_index, 0])
        top = np.polynomial.polynomial.polyval(x, edge_coefficients[order_index, 1])
        slit_pixels = int(info.oversample * np.rint(np.nanmin(top - bottom)))
        y_arc = np.linspace(0.0, info.slit_height_arcsec, slit_pixels)
        pixel_y = (
            bottom[None, :] + (y_arc[:, None] / info.slit_height_arcsec) * (top - bottom)[None, :]
        )
        pixel_x = np.broadcast_to(x[None, :], pixel_y.shape)
        rectified = ndimage.map_coordinates(
            flat,
            [pixel_y, pixel_x],
            order=3,
            mode="nearest",
            prefilter=True,
        )
        rectified = ndimage.median_filter(rectified, size=5, mode="nearest")
        smooth = _smooth_fiterpolate(rectified, info.norm_nxgrid, info.norm_nygrid)
        order_values: list[np.ndarray] = []
        for column_index, column in enumerate(x):
            low = int(np.rint(bottom[column_index] + info.ybuffer))
            high = int(np.rint(top[column_index] - info.ybuffer))
            if low < 0 or high >= flat.shape[0] or high < low:
                continue
            rows = np.arange(low, high + 1)
            arc = (
                (rows - bottom[column_index])
                * info.slit_height_arcsec
                / (top[column_index] - bottom[column_index])
            )
            model = np.interp(arc, y_arc, smooth[:, column_index])
            valid = np.isfinite(model) & (model != 0)
            output = np.ones_like(model)
            output[valid] = flat[rows[valid], column] / model[valid]
            normalized[rows, column] = output
            # SpeXTool divides IVAR by the model (not model squared) here.
            output_variance[rows[valid], column] = variance[rows[valid], column] / model[valid]
            model_image[rows[valid], column] = model[valid]
            order_values.append(output[valid])
        finite_values = np.concatenate(order_values) if order_values else np.array([])
        if finite_values.size:
            center = np.median(finite_values)
            mad = 1.4826 * np.median(np.abs(finite_values - center))
            clipped = (
                finite_values[np.abs(finite_values - center) <= 5 * mad]
                if mad > 0
                else finite_values
            )
            order_rms[order_index] = np.std(clipped)
        else:
            order_rms[order_index] = np.nan
    normalized[~np.isfinite(normalized) | (normalized <= 0)] = 1.0
    return normalized, output_variance, model_image, order_rms


def build_normalized_flat(
    files: Sequence[str | Path],
    detector_calibration: IShellDetectorCalibration,
    flat_info: FlatInfo,
) -> NormalizedFlat:
    """Build the supported SpeXTool iSHELL normalized-flat product.

    Behavioral references are ``mc_ishellcals2dxd.pro``,
    ``mc_scaleimgs.pro``, ``mc_medcomb.pro``, ``mc_adjustguesspos.pro``,
    ``mc_findorders.pro``, and ``mc_normspecflat.pro``.
    """

    paths = tuple(Path(path) for path in files)
    if not paths:
        raise ValueError("at least one flat frame is required")
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for path in paths:
        processed = process_raw_frame(read_ishell_raw(path), detector_calibration)
        if processed.metadata.mode != flat_info.mode:
            raise ValueError(
                f"flat {path.name} is {processed.metadata.mode}, expected {flat_info.mode}"
            )
        images.append(rotate_to_processing(processed.image, flat_info.rotation))
        masks.append(rotate_to_processing(processed.mask, flat_info.rotation))
    cube = np.stack(images)
    mask_cube = np.stack(masks)
    cube[(mask_cube != 0) | ~np.isfinite(cube)] = np.nan
    medians = np.nanmedian(cube, axis=(1, 2))
    common_median = np.median(medians)
    scales = common_median / medians
    cube *= scales[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combined = np.nanmedian(cube, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(cube - combined), axis=0)
    variance = mad**2 / len(paths)
    mask = np.bitwise_or.reduce(mask_cube, axis=0)
    interpolation_flat = _fill_invalid_nearest(combined)
    vertical_offset = find_vertical_order_offset(interpolation_flat, flat_info)
    edge_coefficients, xranges = find_order_edges(interpolation_flat, flat_info, vertical_offset)
    normalized, output_variance, model, order_rms = normalize_spectral_flat(
        interpolation_flat,
        variance,
        edge_coefficients,
        xranges,
        flat_info,
    )
    return NormalizedFlat(
        image=normalized,
        variance=output_variance,
        mask=mask,
        combined_image=combined,
        model=model,
        orders=flat_info.orders.copy(),
        xranges=xranges,
        edge_coefficients=edge_coefficients,
        order_rms=order_rms,
        vertical_offset=vertical_offset,
        input_files=paths,
        mode=flat_info.mode,
        rotation=flat_info.rotation,
        plate_scale_arcsec_per_pixel=flat_info.plate_scale_arcsec_per_pixel,
        slit_height_arcsec=flat_info.slit_height_arcsec,
    )


def write_normalized_flat(product: NormalizedFlat, path: str | Path, *, calib_id: str) -> Path:
    """Write a normalized flat and measured order geometry to FITS."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = "0.1.0.dev0"
    header["STAGE"] = "NORMALIZED_FLAT"
    header["CALIBID"] = calib_id
    header["MODE"] = product.mode
    header["ROTATION"] = product.rotation
    header["ORDERS"] = ",".join(str(value) for value in product.orders)
    header["PLTSCALE"] = product.plate_scale_arcsec_per_pixel
    header["SLTH_ARC"] = product.slit_height_arcsec
    header["YOFFSET"] = product.vertical_offset
    header["NINPUTS"] = len(product.input_files)
    for source in product.input_files:
        header.add_history(f"INPUT {source.name}")
    n_coefficients = product.edge_coefficients.shape[2]
    geometry_columns = [
        fits.Column(name="ORDER", format="I", array=product.orders),
        fits.Column(name="XRANGE", format="2J", array=product.xranges),
        fits.Column(
            name="BOTTOM_COEFF",
            format=f"{n_coefficients}D",
            array=product.edge_coefficients[:, 0],
        ),
        fits.Column(
            name="TOP_COEFF",
            format=f"{n_coefficients}D",
            array=product.edge_coefficients[:, 1],
        ),
        fits.Column(name="RMS", format="D", array=product.order_rms),
    ]
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(
                unrotate_from_processing(product.image, product.rotation).astype(np.float32),
                header,
            ),
            fits.ImageHDU(
                unrotate_from_processing(product.variance, product.rotation).astype(np.float32),
                name="VARIANCE",
            ),
            fits.ImageHDU(
                unrotate_from_processing(product.mask, product.rotation).astype(np.uint8),
                name="MASK",
            ),
            fits.ImageHDU(
                unrotate_from_processing(product.combined_image, product.rotation).astype(
                    np.float32
                ),
                name="COMBINED",
            ),
            fits.ImageHDU(
                unrotate_from_processing(product.model, product.rotation).astype(np.float32),
                name="MODEL",
            ),
            fits.BinTableHDU.from_columns(geometry_columns, name="ORDER_GEOMETRY"),
        ]
    )
    hdus.writeto(output, overwrite=True, checksum=True)
    return output
