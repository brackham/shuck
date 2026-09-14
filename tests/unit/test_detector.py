from datetime import date, time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import shuck.detector as detector
from shuck.detector import (
    ISHELL_BAD_PIXEL_BIT,
    ISHELL_SATURATION_BIT,
    IShellDetectorCalibration,
    apply_amplifier_correction,
    apply_nonlinearity_correction,
    process_raw_frame,
)
from shuck.io import IShellRawMetadata, RawIShellFrame


def test_amplifier_correction_uses_each_regions_bottom_four_rows() -> None:
    image = np.zeros((2048, 2048), dtype=np.float64)
    image[:, :64] = 101.0
    image[2044:, :64] = 11.0
    image[:, 64:128] = 202.0
    image[2044:, 64:128] = 22.0

    corrected = apply_amplifier_correction(image)

    assert corrected[100, 10] == 90.0
    assert corrected[100, 70] == 180.0
    assert np.median(corrected[2044:, :64]) == 0.0
    assert np.median(corrected[2044:, 64:128]) == 0.0


def test_nonlinearity_correction_uses_increasing_polynomial_order() -> None:
    image = np.full((3, 3), 2.0)
    coefficients = np.stack(
        [np.full_like(image, 1.0), np.full_like(image, 0.5), np.full_like(image, 0.25)]
    )
    saturated = np.zeros_like(image, dtype=bool)
    saturated[1, 1] = True

    corrected = apply_nonlinearity_correction(image, coefficients, saturated)

    # Polynomial correction is 1 + 0.5*2 + 0.25*2^2 = 3.
    np.testing.assert_allclose(corrected[~saturated], 2.0 / 3.0)
    assert corrected[1, 1] == 2.0


def test_process_raw_frame_propagates_rate_variance_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (8, 8)
    monkeypatch.setattr(detector, "ISHELL_DETECTOR_SHAPE", shape)
    raw = np.full(shape, 20.0)
    pedestal = np.full(shape, 40_100.0)
    signal = np.full(shape, 40_080.0)
    pedestal[1, 1] = 5.0
    good = np.ones(shape, dtype=bool)
    good[2, 2] = False
    coefficients = np.empty((3, *shape), dtype=np.float32)
    coefficients[0] = 0.0
    coefficients[1] = 1.0
    coefficients[2] = 1.0
    calibration = IShellDetectorCalibration(
        bias_dn=np.full(shape, 40_000.0),
        linearity_limits_and_coefficients=coefficients,
        bad_pixel_good=good,
        hot_pixel_good=np.ones(shape, dtype=bool),
        source_directory=Path("calibration"),
    )
    metadata = IShellRawMetadata(
        itime=10.0,
        coadds=1,
        ndr=2,
        table_se=0.5,
        divisor=2.0,
        mode="J3",
        filename="synthetic.fits",
        date_obs=date(2026, 4, 6),
        time_obs=time(1, 2, 3),
        mjd_obs=61136.0,
        object_name="Synthetic",
        beam="A",
        ra="12:34:56",
        dec="-01:02:03",
        airmass=1.2,
        hour_angle="+00:00:00",
        position_angle=0.0,
    )
    frame = RawIShellFrame(
        path=Path("synthetic.fits"),
        raw_image=raw,
        primary_difference_image=raw.copy(),
        pedestal_image=pedestal,
        signal_image=signal,
        header=fits.Header(),
        metadata=metadata,
        variance=np.ones(shape),
        mask=np.zeros(shape, dtype=np.uint8),
    )

    result = process_raw_frame(
        frame,
        calibration,
        amplifier_correction=False,
        linearity_correction=False,
    )

    assert result.image[5, 5] == 2.0
    assert result.variance.shape == shape
    assert np.all(np.isfinite(result.variance))
    assert np.all(result.variance >= 0)
    assert result.mask[1, 1] & ISHELL_SATURATION_BIT
    assert result.mask[2, 2] & ISHELL_BAD_PIXEL_BIT
    assert result.image_unit == "DN/s"
    assert result.variance_unit == "(DN/s)^2"
