"""Science and standard exposure preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shuck.calibration.dark import MasterDark
from shuck.calibration.flat import NormalizedFlat, _fill_invalid_nearest, rotate_to_processing
from shuck.calibration.rectify import DistortionSolution, RectifiedOrder, rectify_order
from shuck.detector import IShellDetectorCalibration, process_raw_frame
from shuck.io import IShellRawMetadata, read_ishell_raw
from shuck.statistics import robust_weighted_mean


@dataclass(frozen=True)
class PreprocessedExposure:
    """One A-beam object exposure after dark, flat, and rectification."""

    orders: tuple[RectifiedOrder, ...]
    metadata: IShellRawMetadata
    source_path: Path
    dark_files: tuple[Path, ...]
    flat_files: tuple[Path, ...]


def preprocess_exposure(
    path: str | Path,
    detector_calibration: IShellDetectorCalibration,
    master_dark: MasterDark,
    normalized_flat: NormalizedFlat,
    distortion: DistortionSolution,
) -> PreprocessedExposure:
    """Dark-subtract, flat-field, and rectify one iSHELL A exposure.

    The behavioral reference is the ``A-Sky/Dark`` path in SpeXTool 5.0.3
    ``xspextool.pro``. As in that path, flat-field variance is not added to
    the object variance.
    """

    source = Path(path)
    processed = process_raw_frame(read_ishell_raw(source), detector_calibration)
    if processed.metadata.mode != normalized_flat.mode:
        raise ValueError(
            f"exposure mode {processed.metadata.mode!r} does not match flat mode "
            f"{normalized_flat.mode!r}"
        )
    if processed.metadata.beam != "A":
        raise ValueError("v0.1 preprocessing accepts A-beam object exposures only")
    configuration = (
        processed.metadata.itime,
        processed.metadata.ndr,
        processed.metadata.coadds,
    )
    if (
        master_dark.detector_configuration is not None
        and configuration != master_dark.detector_configuration
    ):
        raise ValueError(
            "science exposure and master dark detector configurations differ: "
            f"{configuration} != {master_dark.detector_configuration}"
        )
    if processed.image.shape != master_dark.image.shape:
        raise ValueError("science exposure and master dark shapes differ")

    with np.errstate(divide="ignore", invalid="ignore"):
        dark_subtracted = processed.image - master_dark.image
        variance = processed.variance + master_dark.variance
    image = rotate_to_processing(dark_subtracted, normalized_flat.rotation)
    variance = rotate_to_processing(variance, normalized_flat.rotation)
    detector_mask = rotate_to_processing(processed.mask, normalized_flat.rotation)
    dark_mask = rotate_to_processing(master_dark.mask, normalized_flat.rotation)
    mask = (
        detector_mask.astype(np.uint16)
        | dark_mask.astype(np.uint16)
        | normalized_flat.mask.astype(np.uint16)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        image = image / normalized_flat.image
        variance = variance / normalized_flat.image**2
    invalid = ~np.isfinite(image) | ~np.isfinite(variance) | (variance <= 0)
    mask[invalid] |= np.uint16(1 << 15)

    # Interpolation must see finite values; the propagated mask still rejects
    # every rectified sample whose bilinear footprint touches a repaired pixel.
    interpolation_image = _fill_invalid_nearest(image)
    interpolation_variance = _fill_invalid_nearest(np.where(variance > 0, variance, np.nan))
    rectified = tuple(
        rectify_order(interpolation_image, interpolation_variance, mask, geometry)
        for geometry in distortion.geometries
    )
    return PreprocessedExposure(
        orders=rectified,
        metadata=processed.metadata,
        source_path=source,
        dark_files=master_dark.input_files,
        flat_files=normalized_flat.input_files,
    )


def combine_preprocessed_exposures(
    exposures: tuple[PreprocessedExposure, ...],
    *,
    sigma_clip: float = 8.0,
) -> PreprocessedExposure:
    """Build a robust group image for aperture finding and tracing.

    This applies the same robust inverse-variance combination semantics as
    SpeXTool ``mc_meancomb.pro`` independently at every rectified pixel.  The
    result is used only to derive a common extraction model; spectra remain
    extracted and combined as separate exposures.
    """

    if not exposures:
        raise ValueError("at least one preprocessed exposure is required")
    order_numbers = tuple(order.order for order in exposures[0].orders)
    for exposure in exposures[1:]:
        if tuple(order.order for order in exposure.orders) != order_numbers:
            raise ValueError("preprocessed exposures have different order sets")
    combined_orders: list[RectifiedOrder] = []
    for order_index, order_number in enumerate(order_numbers):
        reference = exposures[0].orders[order_index]
        images = np.stack([exposure.orders[order_index].image for exposure in exposures])
        variances = np.stack([exposure.orders[order_index].variance for exposure in exposures])
        masks = np.stack([exposure.orders[order_index].mask for exposure in exposures])
        combined = robust_weighted_mean(
            images,
            variances,
            axis=0,
            sigma=sigma_clip,
            mask=masks == 0,
        )
        output_mask = np.bitwise_or.reduce(
            np.where(combined.good, masks, 0).astype(np.uint16), axis=0
        )
        invalid = ~np.isfinite(combined.mean) | ~np.isfinite(combined.variance)
        output_mask[invalid] |= np.uint16(1 << 15)
        combined_orders.append(
            RectifiedOrder(
                order=order_number,
                wavelength_micron=reference.wavelength_micron.copy(),
                spatial_arcsec=reference.spatial_arcsec.copy(),
                image=combined.mean,
                variance=combined.variance,
                mask=output_mask,
            )
        )
    return PreprocessedExposure(
        orders=tuple(combined_orders),
        metadata=exposures[0].metadata,
        source_path=Path(f"group_stack_{len(exposures)}_exposures"),
        dark_files=tuple(path for exposure in exposures for path in exposure.dark_files),
        flat_files=exposures[0].flat_files,
    )
