"""Non-interactive QA products for shuck reductions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from shuck.calibration.flat import NormalizedFlat
from shuck.calibration.rectify import DistortionSolution, rectify_order
from shuck.calibration.wavecal import WavelengthSolution
from shuck.combine import CombinedSpectrum
from shuck.extraction.optimal import ExtractedExposure
from shuck.merge import MergedSpectrum
from shuck.telluric import TelluricCorrectedSpectrum, TelluricCorrection

_matplotlib_cache = Path(tempfile.gettempdir()) / "shuck-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def write_flat_qa(product: NormalizedFlat, output_prefix: str | Path) -> tuple[Path, Path]:
    """Write visual and numerical QA for a normalized spectral flat.

    The panels serve the same scientific checks as the flat displays in
    SpeXTool 5.0.3 ``mc_ishellcals2dxd.pro``: order placement, normalization
    structure, and per-order residual scatter.
    """

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")

    finite_combined = product.combined_image[np.isfinite(product.combined_image)]
    lower, upper = np.percentile(finite_combined, [5, 99.5])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].imshow(
        product.combined_image,
        origin="lower",
        cmap="gray",
        vmin=lower,
        vmax=upper,
        aspect="auto",
    )
    for order_index, order in enumerate(product.orders):
        x = np.arange(product.xranges[order_index, 0], product.xranges[order_index, 1] + 1)
        for edge in range(2):
            y = np.polynomial.polynomial.polyval(x, product.edge_coefficients[order_index, edge])
            axes[0].plot(x, y, color="tab:red", linewidth=0.35)
        center_x = x[len(x) // 2]
        center_y = np.mean(
            [
                np.polynomial.polynomial.polyval(
                    center_x, product.edge_coefficients[order_index, edge]
                )
                for edge in range(2)
            ]
        )
        axes[0].text(center_x, center_y, str(order), color="cyan", fontsize=5)
    axes[0].set(title="Combined flat and measured edges", xlabel="Column", ylabel="Row")

    axes[1].imshow(
        product.image,
        origin="lower",
        cmap="coolwarm",
        vmin=0.8,
        vmax=1.2,
        aspect="auto",
    )
    axes[1].set(title="Normalized flat", xlabel="Column", ylabel="Row")
    axes[2].plot(product.orders, product.order_rms, marker="o", markersize=3)
    axes[2].set(
        title="Robust normalized-flat scatter",
        xlabel="Order",
        ylabel="Standard deviation",
    )
    axes[2].grid(alpha=0.25)
    figure.suptitle(
        f"{product.mode} flat: offset={product.vertical_offset:+d} px, "
        f"{len(product.input_files)} inputs"
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    in_order: list[np.ndarray] = []
    for index in range(len(product.orders)):
        x = np.arange(product.xranges[index, 0], product.xranges[index, 1] + 1)
        bottom = np.polynomial.polynomial.polyval(x, product.edge_coefficients[index, 0])
        top = np.polynomial.polynomial.polyval(x, product.edge_coefficients[index, 1])
        for column_index, column in enumerate(x):
            low = max(0, int(np.rint(bottom[column_index] + 1)))
            high = min(product.image.shape[0], int(np.rint(top[column_index] - 1)) + 1)
            in_order.append(product.image[low:high, column])
    values = np.concatenate(in_order)
    values = values[np.isfinite(values)]
    metrics = {
        "mode": product.mode,
        "input_files": [path.name for path in product.input_files],
        "orders": [int(value) for value in product.orders],
        "vertical_offset_pixels": product.vertical_offset,
        "in_order_percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(
                (0, 1, 50, 99, 100), np.percentile(values, (0, 1, 50, 99, 100)), strict=True
            )
        },
        "order_rms": {
            str(int(order)): float(rms)
            for order, rms in zip(product.orders, product.order_rms, strict=True)
        },
        "flagged_fraction": float(np.mean(product.mask != 0)),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_wavecal_qa(product: WavelengthSolution, output_prefix: str | Path) -> tuple[Path, Path]:
    """Write SpeXTool-purpose-equivalent 1DXD residual QA and metrics."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    lines = product.line_measurements
    orders = np.array([line.order for line in lines])
    pixels = np.array([line.fitted_position for line in lines])
    residuals = np.array([line.residual_angstrom for line in lines])
    found = np.array([line.found for line in lines])
    used = np.array([line.used for line in lines])

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for axis, x, label in (
        (axes[0], orders, "Order"),
        (axes[1], pixels, "Detector column"),
    ):
        axis.scatter(x[found & ~used], residuals[found & ~used], marker="x", color="tab:red")
        axis.scatter(x[used], residuals[used], s=16, color="tab:blue")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(product.rms_angstrom, color="gray", linestyle="--", linewidth=0.8)
        axis.axhline(-product.rms_angstrom, color="gray", linestyle="--", linewidth=0.8)
        axis.set(xlabel=label, ylabel="Data − model (Å)")
        axis.grid(alpha=0.2)
    axes[0].set_title(
        f"{product.mode} 1DXD: RMS={product.rms_angstrom:.4f} Å, "
        f"offset={product.cross_correlation_offset:+.3f} px"
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    accepted = residuals[used]
    metrics = {
        "mode": product.mode,
        "input_files": [path.name for path in product.input_files],
        "cross_correlation_offset_pixels": product.cross_correlation_offset,
        "rms_angstrom": product.rms_angstrom,
        "line_count": len(lines),
        "found_count": int(np.count_nonzero(found)),
        "used_count": int(np.count_nonzero(used)),
        "accepted_residual_percentiles_angstrom": {
            str(percentile): float(value)
            for percentile, value in zip(
                (0, 16, 50, 84, 100),
                np.percentile(accepted, (0, 16, 50, 84, 100)),
                strict=True,
            )
        },
        "wavelength_ranges_micron": {
            str(int(order)): [
                float(product.wavelength(int(order), np.array([xrange[0]]))[0]),
                float(product.wavelength(int(order), np.array([xrange[1]]))[0]),
            ]
            for order, xrange in zip(product.orders, product.xranges, strict=True)
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_rectification_qa(
    product: DistortionSolution,
    wavecal: WavelengthSolution,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Write line-trace, distortion-fit, and rectified-arc QA."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    x_mid = np.array([trace.x_mid for trace in product.traces])
    y_mid = np.array([trace.y_mid for trace in product.traces])
    slope = np.array([trace.slope for trace in product.traces])
    modeled = np.polynomial.polynomial.polyval2d(
        x_mid,
        y_mid,
        product.slope_coefficients.reshape(
            product.slope_y_degree + 1,
            product.slope_x_degree + 1,
        ).T,
    )
    residual = slope - modeled
    middle_geometry = product.geometries[len(product.geometries) // 2]
    rectified = rectify_order(
        np.where(np.isfinite(wavecal.arc_image), wavecal.arc_image, 0.0),
        np.where(np.isfinite(wavecal.arc_variance), wavecal.arc_variance, 1.0),
        np.zeros_like(wavecal.arc_image, dtype=np.uint16),
        middle_geometry,
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    finite_arc = wavecal.arc_image[np.isfinite(wavecal.arc_image)]
    lower, upper = np.percentile(finite_arc, (10, 99.8))
    axes[0].imshow(
        wavecal.arc_image,
        origin="lower",
        cmap="gray",
        vmin=lower,
        vmax=upper,
        aspect="auto",
    )
    for trace in product.traces:
        axes[0].plot(trace.x, trace.y, color="tab:red", linewidth=0.25, alpha=0.7)
    axes[0].set(title="Measured 2-D line traces", xlabel="Column", ylabel="Row")
    colors = np.where(product.slope_fit_used, "tab:blue", "tab:red")
    axes[1].scatter(x_mid, residual, c=colors, s=15)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Slope-surface residuals", xlabel="Detector column", ylabel="Slope residual")
    arc_values = rectified.image[np.isfinite(rectified.image)]
    arc_lower, arc_upper = np.percentile(arc_values, (10, 99.8))
    axes[2].imshow(
        rectified.image,
        origin="lower",
        cmap="gray",
        vmin=arc_lower,
        vmax=arc_upper,
        aspect="auto",
        extent=(
            rectified.wavelength_micron[0],
            rectified.wavelength_micron[-1],
            rectified.spatial_arcsec[0],
            rectified.spatial_arcsec[-1],
        ),
    )
    axes[2].set(
        title=f"Rectified arc, order {rectified.order}",
        xlabel="Wavelength (µm)",
        ylabel="Slit position (arcsec)",
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    trace_rms = np.array([trace.rms_pixels for trace in product.traces])
    metrics = {
        "mode": wavecal.mode,
        "trace_count": len(product.traces),
        "slope_fit_used_count": int(np.count_nonzero(product.slope_fit_used)),
        "trace_rms_pixel_percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(
                (0, 25, 50, 75, 100),
                np.percentile(trace_rms, (0, 25, 50, 75, 100)),
                strict=True,
            )
        },
        "slope_residual_rms": float(np.sqrt(np.mean(residual[product.slope_fit_used] ** 2))),
        "slope_coefficients": product.slope_coefficients.tolist(),
        "rectified_shapes": {
            str(geometry.order): list(geometry.x_index.shape) for geometry in product.geometries
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_extraction_qa(
    product: ExtractedExposure,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Write aperture, trace, and extracted-spectrum QA for one exposure."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    orders = np.array([aperture.order for aperture in product.apertures])
    positions = np.array([aperture.position_arcsec for aperture in product.apertures])
    fwhm = np.array([aperture.fwhm_arcsec for aperture in product.apertures])
    trace_rms = np.array([trace.rms_arcsec for trace in product.traces])

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    profile_grid = np.stack([profile.normalized_flux for profile in product.profiles])
    spatial = product.profiles[0].spatial_arcsec
    profile_limit = np.nanpercentile(np.abs(profile_grid), 99)
    axes[0, 0].imshow(
        profile_grid,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-profile_limit,
        vmax=profile_limit,
        extent=(spatial[0], spatial[-1], orders[0] - 0.5, orders[-1] + 0.5),
    )
    axes[0, 0].plot(positions, orders, color="black", marker="o", markersize=2, linewidth=0.7)
    axes[0, 0].set(
        xlabel="Slit position (arcsec)",
        ylabel="Order",
        title="Spatial profiles and apertures",
    )
    axes[0, 1].plot(orders, fwhm, marker="o", markersize=3)
    axes[0, 1].set(xlabel="Order", ylabel="FWHM (arcsec)", title="Spatial-profile width")
    axes[1, 0].semilogy(orders, trace_rms, marker="o", markersize=3)
    axes[1, 0].set(xlabel="Order", ylabel="RMS (arcsec)", title="Trace residuals")
    for order in product.orders:
        finite = np.isfinite(order.flux)
        if not np.any(finite):
            continue
        scale = np.nanmedian(order.flux[finite])
        if scale != 0:
            axes[1, 1].plot(
                order.wavelength_micron[finite],
                order.flux[finite] / scale,
                linewidth=0.5,
            )
    axes[1, 1].set(
        xlabel="Wavelength (µm)",
        ylabel="Median-normalized flux",
        title="Extracted orders",
    )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    figure.suptitle(product.source_path.name)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    finite_fraction = {
        str(order.order): float(np.mean(np.isfinite(order.flux))) for order in product.orders
    }
    median_snr = {
        str(order.order): float(np.nanmedian(order.flux / order.uncertainty))
        for order in product.orders
    }
    metrics = {
        "input_file": product.source_path.name,
        "order_count": len(product.orders),
        "valid_aperture_count": sum(aperture.valid for aperture in product.apertures),
        "valid_trace_count": sum(trace.valid for trace in product.traces),
        "aperture_position_arcsec": {
            str(aperture.order): aperture.position_arcsec for aperture in product.apertures
        },
        "aperture_fwhm_arcsec": {
            str(aperture.order): aperture.fwhm_arcsec for aperture in product.apertures
        },
        "trace_rms_arcsec": {str(trace.order): trace.rms_arcsec for trace in product.traces},
        "finite_flux_fraction": finite_fraction,
        "median_signal_to_noise": median_snr,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_combination_qa(
    product: CombinedSpectrum,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Write combined-order spectra, signal-to-noise, and scaling QA."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    median_snr: dict[str, float] = {}
    finite_fraction: dict[str, float] = {}
    for order in product.orders:
        good = (order.mask == 0) & np.isfinite(order.flux) & np.isfinite(order.uncertainty)
        finite_fraction[str(order.order)] = float(np.mean(good))
        if not np.any(good):
            continue
        scale = np.nanmedian(order.flux[good])
        axes[0].plot(
            order.wavelength_micron[good],
            order.flux[good] / scale,
            linewidth=0.5,
        )
        snr = order.flux[good] / order.uncertainty[good]
        median_snr[str(order.order)] = float(np.nanmedian(snr))
        axes[1].plot(order.wavelength_micron[good], snr, linewidth=0.5)
    axes[0].set(ylabel="Median-normalized flux", title="Combined orders")
    axes[1].set(xlabel="Wavelength (µm)", ylabel="Signal-to-noise")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{len(product.input_files)} spectra; scale order {product.scale_order}")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    metrics = {
        "input_files": [path.name for path in product.input_files],
        "scale_order": product.scale_order,
        "scale_factors": product.scale_factors.tolist(),
        "sigma_clip": product.sigma_clip,
        "finite_fraction": finite_fraction,
        "median_signal_to_noise": median_snr,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_telluric_qa(
    correction: TelluricCorrection,
    corrected: TelluricCorrectedSpectrum,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Write correction-shape and corrected-order QA for ``xtellcor`` output."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    finite_fraction: dict[str, float] = {}
    standard_finite_fraction: dict[str, float] = {}
    median_snr: dict[str, float] = {}
    correction_by_order = {order.order: order for order in correction.orders}
    standard_by_order = {order.order: order for order in correction.standard_orders}
    vega_by_order = {order.order: order for order in correction.vega_orders}
    for order in corrected.orders:
        telluric_order = correction_by_order[order.order]
        standard_order = standard_by_order[order.order]
        vega_order = vega_by_order[order.order]
        standard_good = (
            (standard_order.mask == 0)
            & np.isfinite(standard_order.flux)
            & (standard_order.flux > 0)
        )
        vega_good = np.isfinite(vega_order.flux) & (vega_order.flux > 0)
        standard_finite_fraction[str(order.order)] = float(np.mean(standard_good))
        if np.any(standard_good) and np.any(vega_good):
            standard_scale = np.nanmedian(standard_order.flux[standard_good])
            vega_scale = np.nanmedian(vega_order.flux[vega_good])
            axes[0, 0].plot(
                standard_order.wavelength_micron[standard_good],
                standard_order.flux[standard_good] / standard_scale,
                color="tab:blue",
                linewidth=0.35,
            )
            axes[0, 0].plot(
                vega_order.wavelength_micron[vega_good],
                vega_order.flux[vega_good] / vega_scale,
                color="tab:orange",
                linewidth=0.35,
            )
        telluric_good = (telluric_order.mask == 0) & np.isfinite(telluric_order.flux)
        axes[0, 1].plot(
            telluric_order.wavelength_micron[telluric_good],
            telluric_order.flux[telluric_good],
            linewidth=0.5,
        )
        good = (order.mask == 0) & np.isfinite(order.flux) & np.isfinite(order.uncertainty)
        finite_fraction[str(order.order)] = float(np.mean(good))
        if not np.any(good):
            continue
        scale = np.nanmedian(order.flux[good])
        if scale != 0:
            axes[1, 0].plot(order.wavelength_micron[good], order.flux[good] / scale, linewidth=0.5)
        axes[1, 1].plot(
            order.wavelength_micron[good],
            order.flux[good] / order.uncertainty[good],
            linewidth=0.5,
        )
        median_snr[str(order.order)] = float(
            np.nanmedian(order.flux[good] / order.uncertainty[good])
        )
    axes[0, 0].plot([], [], color="tab:blue", label="Observed A0V")
    axes[0, 0].plot([], [], color="tab:orange", label="Modified Vega")
    axes[0, 0].legend(loc="best")
    axes[0, 0].set(
        ylabel="Per-order normalized flux",
        title="Observed A0V and modified Vega",
    )
    axes[0, 1].set(ylabel="Correction multiplier", title="Derived telluric correction")
    axes[1, 0].set(
        xlabel="Wavelength (µm)",
        ylabel="Median-normalized flux",
        title="Corrected science orders",
    )
    axes[1, 1].set(
        xlabel="Wavelength (µm)",
        ylabel="Signal-to-noise",
        title="Corrected science signal-to-noise",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"{corrected.science_group} via {corrected.standard_group}; "
        f"Vega shift={correction.applied_velocity_shift_kms:+.3f} km/s"
    )
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    metrics = {
        "science_group": corrected.science_group,
        "standard_group": corrected.standard_group,
        "standard_b_magnitude": correction.b_magnitude,
        "standard_v_magnitude": correction.v_magnitude,
        "standard_catalog_rv_kms": correction.catalog_rv_kms,
        "earth_lsr_velocity_kms": correction.earth_lsr_velocity_kms,
        "applied_velocity_shift_kms": correction.applied_velocity_shift_kms,
        "science_airmass": corrected.science_airmass,
        "standard_airmass": corrected.standard_airmass,
        "airmass_difference": (
            None
            if corrected.science_airmass is None or corrected.standard_airmass is None
            else corrected.standard_airmass - corrected.science_airmass
        ),
        "standard_finite_fraction": standard_finite_fraction,
        "finite_fraction": finite_fraction,
        "median_signal_to_noise": median_snr,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path


def write_merge_qa(product: MergedSpectrum, output_prefix: str | Path) -> tuple[Path, Path]:
    """Write continuous-spectrum and overlap QA for ``mc_mergespec`` output."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_path = prefix.with_suffix(".png")
    metrics_path = prefix.with_suffix(".json")
    good = (
        (product.mask == 0)
        & np.isfinite(product.flux)
        & np.isfinite(product.uncertainty)
        & (product.uncertainty > 0)
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(product.wavelength_micron[good], product.flux[good], linewidth=0.5)
    axes[1].plot(
        product.wavelength_micron[good],
        product.flux[good] / product.uncertainty[good],
        linewidth=0.5,
    )
    for overlap in product.overlap_ranges:
        if overlap is not None:
            for axis in axes:
                axis.axvspan(*overlap, color="tab:blue", alpha=0.04)
    axes[0].set(ylabel="Flux", title="Merged spectrum")
    axes[1].set(xlabel="Wavelength (µm)", ylabel="Signal-to-noise")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    metrics = {
        "orders": list(product.orders),
        "pixel_count": int(product.wavelength_micron.size),
        "finite_unflagged_fraction": float(np.mean(good)),
        "wavelength_range_micron": [
            float(np.nanmin(product.wavelength_micron)),
            float(np.nanmax(product.wavelength_micron)),
        ],
        "overlap_ranges_micron": [
            None if item is None else list(item) for item in product.overlap_ranges
        ],
        "median_signal_to_noise": (
            float(np.nanmedian(product.flux[good] / product.uncertainty[good]))
            if np.any(good)
            else None
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return plot_path, metrics_path
