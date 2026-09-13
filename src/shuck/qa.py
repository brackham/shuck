"""Non-interactive QA products for shuck reductions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from shuck.calibration.flat import NormalizedFlat

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
