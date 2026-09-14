"""iSHELL detector-level corrections, variances, and masks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from shuck.io import IShellRawMetadata, RawIShellFrame

ISHELL_GAIN_ELECTRONS_PER_DN = 1.8
ISHELL_READ_NOISE_ELECTRONS = 10.0
ISHELL_LINEARITY_MAX_DN = 30_000.0
ISHELL_SATURATION_BIT = np.uint8(1)
ISHELL_BAD_PIXEL_BIT = np.uint8(2)
ISHELL_DETECTOR_SHAPE = (2048, 2048)


@dataclass(frozen=True)
class IShellDetectorCalibration:
    """Detector calibration arrays used by SpeXTool's iSHELL read path."""

    bias_dn: np.ndarray
    linearity_limits_and_coefficients: np.ndarray
    bad_pixel_good: np.ndarray
    hot_pixel_good: np.ndarray
    source_directory: Path

    @property
    def linearity_coefficients(self) -> np.ndarray:
        """Polynomial coefficients, ordered from constant to highest power."""

        return self.linearity_limits_and_coefficients[2:]


@dataclass(frozen=True)
class ProcessedDetectorFrame:
    """One SpeXTool-equivalent, detector-corrected iSHELL exposure.

    ``image`` is in DN/s, ``variance`` in (DN/s)^2, and all arrays retain
    detector-native NumPy ``(row, column)`` orientation. Bit zero of ``mask``
    marks samples exceeding the configured linearity maximum; bit one marks
    pixels rejected by either static iSHELL detector mask.
    """

    image: np.ndarray
    variance: np.ndarray
    mask: np.ndarray
    metadata: IShellRawMetadata
    source_path: Path
    image_unit: str = "DN/s"
    variance_unit: str = "(DN/s)^2"


def load_ishell_detector_calibration(
    spextool_directory: str | Path,
) -> IShellDetectorCalibration:
    """Load the detector assets used by SpeXTool 5.0.3 for iSHELL.

    The behavioral references are ``mc_readishellfits.pro`` and the iSHELL
    instrument configuration. ``spextool_directory`` is a Spextool-format
    root containing ``instruments/ishell/data``. Normal pipeline execution
    supplies shuck's installed package-data root; an explicit root remains
    useful for developer comparisons.
    """

    root = Path(spextool_directory).expanduser().resolve()
    directory = root / "instruments" / "ishell" / "data"
    required = {
        "bias": directory / "ishell_bias.fits",
        "linearity": directory / "ishell_lincorr_CDS.fits",
        "bad": directory / "ishell_bdpxmk.fits",
        "hot": directory / "ishell_htpxmk.fits",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SpeXTool iSHELL detector asset(s): " + ", ".join(missing))

    bias, bias_header = fits.getdata(required["bias"], header=True, memmap=False)
    divisor = _positive_finite_float(bias_header.get("DIVISOR"), "bias DIVISOR")
    bias_dn = np.asarray(bias, dtype=np.float64) / divisor
    linearity = np.asarray(fits.getdata(required["linearity"], memmap=False), dtype=np.float64)
    bad = np.asarray(fits.getdata(required["bad"], memmap=False), dtype=bool)
    hot = np.asarray(fits.getdata(required["hot"], memmap=False), dtype=bool)

    if bias_dn.shape != ISHELL_DETECTOR_SHAPE:
        raise ValueError(f"Unexpected iSHELL bias shape {bias_dn.shape}")
    if linearity.ndim != 3 or linearity.shape[1:] != ISHELL_DETECTOR_SHAPE:
        raise ValueError(f"Unexpected iSHELL linearity-cube shape {linearity.shape}")
    if linearity.shape[0] < 3:
        raise ValueError("iSHELL linearity cube must contain fit limits and coefficients")
    for name, array in (("bad-pixel", bad), ("hot-pixel", hot)):
        if array.shape != ISHELL_DETECTOR_SHAPE:
            raise ValueError(f"Unexpected iSHELL {name} mask shape {array.shape}")

    return IShellDetectorCalibration(
        bias_dn=bias_dn,
        linearity_limits_and_coefficients=linearity,
        bad_pixel_good=bad,
        hot_pixel_good=hot,
        source_directory=directory,
    )


def apply_amplifier_correction(image: np.ndarray) -> np.ndarray:
    """Subtract each amplifier's bottom-reference-pixel median.

    This reproduces SpeXTool 5.0.3 ``mc_ishellampcor.pro``: the detector has
    32 contiguous 64-column amplifier regions, and rows 2044--2047 provide
    the reference pixels for each region.
    """

    source = np.asarray(image, dtype=np.float64)
    if source.shape != ISHELL_DETECTOR_SHAPE:
        raise ValueError(f"iSHELL amplifier correction requires shape {ISHELL_DETECTOR_SHAPE}")
    corrected = source.copy()
    for amplifier in range(32):
        start = amplifier * 64
        stop = start + 64
        level = np.median(source[2044:2048, start:stop])
        corrected[:, start:stop] -= level
    return corrected


def apply_nonlinearity_correction(
    image: np.ndarray,
    coefficients: np.ndarray,
    saturation_mask: np.ndarray,
) -> np.ndarray:
    """Apply SpeXTool's per-pixel polynomial nonlinearity correction.

    ``coefficients`` has shape ``(degree + 1, rows, columns)`` in increasing
    polynomial order. Pixels marked saturated, plus the four-pixel detector
    border, receive a unity correction exactly as in the non-pair path of
    ``mc_readishellfits.pro``.
    """

    source = np.asarray(image, dtype=np.float64)
    coeffs = np.asarray(coefficients, dtype=np.float64)
    saturated = np.asarray(saturation_mask, dtype=bool)
    if coeffs.ndim != 3 or coeffs.shape[1:] != source.shape:
        raise ValueError("linearity coefficients must have shape (ncoeff, rows, columns)")
    if coeffs.shape[0] < 1:
        raise ValueError("at least one linearity coefficient is required")
    if saturated.shape != source.shape:
        raise ValueError("saturation mask shape does not match image")

    correction = coeffs[-1].copy()
    for coefficient in coeffs[-2::-1]:
        correction = correction * source + coefficient
    correction[saturated] = 1.0
    if source.shape[0] >= 8 and source.shape[1] >= 8:
        correction[:4, :] = 1.0
        correction[-4:, :] = 1.0
        correction[:, :4] = 1.0
        correction[:, -4:] = 1.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return source / correction


def process_raw_frame(
    frame: RawIShellFrame,
    calibration: IShellDetectorCalibration,
    *,
    amplifier_correction: bool = True,
    linearity_correction: bool = True,
    linearity_max_dn: float = ISHELL_LINEARITY_MAX_DN,
) -> ProcessedDetectorFrame:
    """Apply the supported SpeXTool iSHELL detector processing path.

    The behavioral reference is the non-pair branch of SpeXTool 5.0.3
    ``mc_readishellfits.pro`` with the iSHELL defaults ``AMPCOR=1``,
    ``LINCOR=1``, and ``LINCORMAX=30000``. Variance follows that routine's
    multiple-read shot-plus-read-noise expression and intentionally does not
    add uncertainty for the deterministic correction arrays.
    """

    if frame.raw_image.shape != ISHELL_DETECTOR_SHAPE:
        raise ValueError(f"iSHELL processing requires shape {ISHELL_DETECTOR_SHAPE}")
    linearity_max_dn = _positive_finite_float(linearity_max_dn, "linearity_max_dn")

    saturated = (frame.pedestal_image < calibration.bias_dn - linearity_max_dn) | (
        frame.signal_image < calibration.bias_dn - linearity_max_dn
    )
    mask = np.zeros(frame.raw_image.shape, dtype=np.uint8)
    mask[saturated] |= ISHELL_SATURATION_BIT
    static_bad = ~(calibration.bad_pixel_good & calibration.hot_pixel_good)
    mask[static_bad] |= ISHELL_BAD_PIXEL_BIT

    corrected_dn = np.asarray(frame.raw_image, dtype=np.float64)
    if amplifier_correction:
        corrected_dn = apply_amplifier_correction(corrected_dn)
    if linearity_correction:
        corrected_dn = apply_nonlinearity_correction(
            corrected_dn,
            calibration.linearity_coefficients,
            saturated,
        )

    variance_dn2 = initial_uncorrected_variance(
        corrected_dn,
        itime=frame.metadata.itime,
        coadds=frame.metadata.coadds,
        ndr=frame.metadata.ndr,
        table_se=frame.metadata.table_se,
        divisor=frame.metadata.divisor,
    )
    image_rate = corrected_dn / frame.metadata.itime
    variance_rate = variance_dn2 / frame.metadata.itime**2
    return ProcessedDetectorFrame(
        image=image_rate,
        variance=variance_rate,
        mask=mask,
        metadata=frame.metadata,
        source_path=frame.path,
    )


def initial_uncorrected_variance(
    raw_image: np.ndarray,
    *,
    itime: float,
    coadds: int,
    ndr: int,
    table_se: float,
    divisor: float,
    gain: float = ISHELL_GAIN_ELECTRONS_PER_DN,
    read_noise: float = ISHELL_READ_NOISE_ELECTRONS,
) -> np.ndarray:
    """Return the no-correction variance of an iSHELL raw image in DN squared.

    ``raw_image`` is the pedestal-minus-signal image after division by the
    FITS ``DIVISOR``, in DN. ``itime`` and ``table_se`` are in seconds,
    ``gain`` is in electrons per DN, and ``read_noise`` is the noise in
    electrons per single read. The result has the same shape as
    ``raw_image`` and units of DN squared.

    The behavioral reference is SpeXTool 5.0.3
    ``instruments/ishell/pro/mc_readishellfits.pro`` in its non-pair,
    no-amplifier-correction, no-linearity-correction path. SpeXTool returns
    a DN/s image and variance in (DN/s) squared. This helper evaluates the
    same expression before the final exposure-time normalization so its
    output is the variance of Shuck's DN-valued raw image. Dividing the
    result by ``itime**2`` gives the SpeXTool rate variance.
    """

    image = np.asarray(raw_image, dtype=np.float64)
    itime = _positive_finite_float(itime, "itime")
    coadds = _positive_int(coadds, "coadds")
    ndr = _positive_int(ndr, "ndr")
    table_se = _positive_finite_float(table_se, "table_se")
    divisor = _positive_finite_float(divisor, "divisor")
    gain = _positive_finite_float(gain, "gain")
    read_noise = _positive_finite_float(read_noise, "read_noise")

    correction = 1.0 - table_se * (ndr**2 - 1.0) / (3.0 * itime * ndr)
    if not np.isfinite(correction) or correction < 0.0:
        raise ValueError(
            "ITIME, NDR, and TABLE_SE produce a negative or non-finite "
            "multiple-read variance correction"
        )

    shot_variance = np.abs(image) * divisor * correction / ndr / (coadds**2) / gain
    read_variance = 2.0 * read_noise**2 / ndr / coadds / gain**2
    return shot_variance + read_variance


def initialize_detector_mask(shape: Sequence[int]) -> np.ndarray:
    """Return the zeroed uint8 mask used by the raw iSHELL read path.

    The behavioral reference is the initial ``bytarr`` mask in SpeXTool
    5.0.3 ``instruments/ishell/pro/mc_readishellfits.pro``. Detector flags
    are populated only by later correction stages.
    """

    return np.zeros(tuple(shape), dtype=np.uint8)


def _positive_finite_float(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if converted <= 0 or converted != value:
        raise ValueError(f"{name} must be a positive integer")
    return converted
