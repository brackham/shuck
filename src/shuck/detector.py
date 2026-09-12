"""iSHELL detector-level corrections, variances, and masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ISHELL_GAIN_ELECTRONS_PER_DN = 1.8
ISHELL_READ_NOISE_ELECTRONS = 10.0


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
