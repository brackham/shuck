"""Master-dark construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from shuck.detector import IShellDetectorCalibration, process_raw_frame
from shuck.io import read_ishell_raw
from shuck.statistics import robust_weighted_mean


@dataclass(frozen=True)
class MasterDark:
    """A robustly combined master dark in detector-native orientation."""

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    input_files: tuple[Path, ...]
    sigma_clip: float
    rejected_fraction: float
    image_unit: str = "DN/s"
    variance_unit: str = "(DN/s)^2"


def build_master_dark(
    files: Sequence[str | Path],
    calibration: IShellDetectorCalibration,
    *,
    sigma_clip: float = 8.0,
) -> MasterDark:
    """Build a SpeXTool-style robust weighted-mean master dark.

    The behavioral references are SpeXTool 5.0.3 ``mc_readishellfits.pro``,
    ``mc_meancomb.pro``, and the robust-weighted-mean image-combination path
    in ``xspextool.pro``. Inputs must share ``(ITIME, NDR, CO_ADDS)``.
    """

    paths = tuple(Path(path) for path in files)
    if not paths:
        raise ValueError("at least one dark frame is required")

    images: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    configuration: tuple[float, int, int] | None = None
    for path in paths:
        processed = process_raw_frame(read_ishell_raw(path), calibration)
        current = (
            processed.metadata.itime,
            processed.metadata.ndr,
            processed.metadata.coadds,
        )
        if configuration is None:
            configuration = current
        elif current != configuration:
            raise ValueError(
                "master-dark inputs do not share exact (ITIME, NDR, CO_ADDS): "
                f"expected {configuration}, found {current} for {path.name}"
            )
        images.append(processed.image)
        variances.append(processed.variance)
        masks.append(processed.mask)

    image_cube = np.stack(images)
    variance_cube = np.stack(variances)
    mask_cube = np.stack(masks)
    combined = robust_weighted_mean(
        image_cube,
        variance_cube,
        axis=0,
        sigma=sigma_clip,
        mask=mask_cube == 0,
    )
    output_mask = np.bitwise_or.reduce(mask_cube, axis=0)
    output_mask[~np.isfinite(combined.mean)] |= np.uint8(128)
    initially_good = (
        np.isfinite(image_cube)
        & np.isfinite(variance_cube)
        & (variance_cube > 0)
        & (mask_cube == 0)
    )
    denominator = np.count_nonzero(initially_good)
    rejected = np.count_nonzero(initially_good & ~combined.good)
    rejected_fraction = rejected / denominator if denominator else 0.0
    return MasterDark(
        image=combined.mean,
        variance=combined.variance,
        mask=output_mask,
        input_files=paths,
        sigma_clip=sigma_clip,
        rejected_fraction=rejected_fraction,
    )


def write_master_dark(product: MasterDark, path: str | Path, *, dark_id: str) -> Path:
    """Write a master dark and its uncertainty, mask, and provenance to FITS."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["SHUCKVER"] = ("0.1.0.dev0", "shuck software version")
    header["STAGE"] = ("MASTER_DARK", "shuck processing stage")
    header["DARKID"] = (dark_id, "control-file dark identifier")
    header["NINPUTS"] = (len(product.input_files), "number of raw dark frames")
    header["SIGCLIP"] = (product.sigma_clip, "MAD sigma clipping threshold")
    header["REJFRAC"] = (product.rejected_fraction, "fraction of initially good samples rejected")
    header["BUNIT"] = product.image_unit
    for source in product.input_files:
        header.add_history(f"INPUT {source.name}")
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(np.asarray(product.image, dtype=np.float32), header=header),
            fits.ImageHDU(np.asarray(product.variance, dtype=np.float32), name="VARIANCE"),
            fits.ImageHDU(np.asarray(product.mask, dtype=np.uint8), name="MASK"),
        ]
    )
    hdus[1].header["BUNIT"] = product.variance_unit
    hdus.writeto(output, overwrite=True, checksum=True)
    return output
