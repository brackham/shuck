import shutil
from datetime import date, time
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from shuck.detector import initial_uncorrected_variance
from shuck.io import ISHELL_DETECTOR_SHAPE, RawIShellFitsError, read_ishell_raw


def _raw_header() -> fits.Header:
    header = fits.Header(
        {
            "INSTRUME": "iSHELL Spectrograph",
            "ITIME": 12.5,
            "CO_ADDS": 2,
            "NDR": 4,
            "TABLE_SE": 0.5,
            "DIVISOR": 4,
            "XDTILT": "J3",
            "IRAFNAME": "synthetic.00001.a.fits",
            "DATE_OBS": "2026-04-06",
            "TIME_OBS": "01:02:03.456",
            "MJD_OBS": 61136.0430955556,
            "OBJECT": "Synthetic Target",
            "BEAM": "A",
            "TCS_RA": " 12:34:56.70",
            "TCS_DEC": "-10:20:30.4",
            "TCS_AM": 1.25,
            "TCS_HA": "-01:02:03.4",
            "POSANGLE": -74.18,
            "CUSTOM": "preserve me",
        }
    )
    header.add_history("first preserved history entry")
    header.add_history("second preserved history entry")
    return header


@pytest.fixture(scope="module")
def native_mef(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("raw-ishell")
    primary = np.full(ISHELL_DETECTOR_SHAPE, 39, dtype=np.int32)
    pedestal = np.full(ISHELL_DETECTOR_SHAPE, 100, dtype=np.int32)
    signal = np.full(ISHELL_DETECTOR_SHAPE, 60, dtype=np.int32)

    primary[0, 1] = 111
    primary[1, 0] = -222
    pedestal[0, 1] = 180
    signal[1, 0] = 20

    path = directory / "native.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(data=primary, header=_raw_header()),
            fits.ImageHDU(data=pedestal, name="SUM_PED"),
            fits.ImageHDU(data=signal, name="SUM_SAM"),
        ]
    ).writeto(path)
    return path


def test_read_ishell_raw_maps_hdus_reconstructs_sign_and_applies_divisor(
    native_mef: Path,
) -> None:
    frame = read_ishell_raw(native_mef)

    assert frame.primary_difference_image[0, 1] == 111 / 4
    assert frame.primary_difference_image[1, 0] == -222 / 4
    assert frame.pedestal_image[0, 1] == 180 / 4
    assert frame.signal_image[1, 0] == 20 / 4
    assert frame.raw_image[0, 1] == (180 - 60) / 4
    assert frame.raw_image[1, 0] == (100 - 20) / 4
    np.testing.assert_array_equal(frame.raw_image, frame.pedestal_image - frame.signal_image)
    assert frame.image_unit == "DN"
    assert frame.variance_unit == "DN2"


def test_read_ishell_raw_preserves_detector_orientation(native_mef: Path) -> None:
    frame = read_ishell_raw(native_mef)

    assert frame.raw_image.shape == (2048, 2048)
    assert frame.raw_image[0, 1] == 30
    assert frame.raw_image[1, 0] == 20


def test_read_ishell_raw_parses_typed_metadata(native_mef: Path) -> None:
    metadata = read_ishell_raw(native_mef).metadata

    assert metadata.itime == 12.5
    assert metadata.coadds == 2
    assert metadata.ndr == 4
    assert metadata.table_se == 0.5
    assert metadata.divisor == 4.0
    assert metadata.mode == "J3"
    assert metadata.filename == "synthetic.00001.a.fits"
    assert metadata.date_obs == date(2026, 4, 6)
    assert metadata.time_obs == time(1, 2, 3, 456000)
    assert metadata.mjd_obs == 61136.0430955556
    assert metadata.object_name == "Synthetic Target"
    assert metadata.beam == "A"
    assert metadata.ra == "12:34:56.70"
    assert metadata.dec == "-10:20:30.4"
    assert metadata.airmass == 1.25
    assert metadata.hour_angle == "-01:02:03.4"
    assert metadata.position_angle == -74.18


def test_read_ishell_raw_preserves_complete_primary_header(native_mef: Path) -> None:
    frame = read_ishell_raw(native_mef)
    source_header = fits.getheader(native_mef, ext=0)

    assert frame.header.tostring() == source_header.tostring()
    assert frame.header["CUSTOM"] == "preserve me"
    assert list(frame.header["HISTORY"]) == [
        "first preserved history entry",
        "second preserved history entry",
    ]


def test_initial_uncorrected_variance_matches_spextool_no_correction_formula() -> None:
    raw_image = np.array([[-8.0, 0.0], [12.0, 20.0]])
    itime = 12.5
    coadds = 2
    ndr = 4
    table_se = 0.5
    divisor = 4.0
    gain = 1.8
    read_noise = 10.0

    correction = 1.0 - table_se * (ndr**2 - 1.0) / (3.0 * itime * ndr)
    read_variance_rate = 2.0 * read_noise**2 / ndr / coadds / itime**2 / gain**2
    expected_rate_variance = (
        np.abs(raw_image * divisor) * correction / ndr / coadds**2 / itime**2 / gain
        + read_variance_rate
    )

    variance = initial_uncorrected_variance(
        raw_image,
        itime=itime,
        coadds=coadds,
        ndr=ndr,
        table_se=table_se,
        divisor=divisor,
    )

    np.testing.assert_allclose(variance / itime**2, expected_rate_variance, rtol=1e-15)


def test_read_ishell_raw_initializes_zero_uint8_mask(native_mef: Path) -> None:
    frame = read_ishell_raw(native_mef)

    assert frame.mask.shape == frame.raw_image.shape
    assert frame.mask.dtype == np.uint8
    assert np.count_nonzero(frame.mask) == 0


def test_read_ishell_raw_rejects_malformed_hdu_structure(tmp_path: Path) -> None:
    path = tmp_path / "two_hdus.fits"
    image = np.zeros((2, 2), dtype=np.int16)
    fits.HDUList([fits.PrimaryHDU(data=image), fits.ImageHDU(data=image)]).writeto(path)

    with pytest.raises(RawIShellFitsError, match="exactly 3 HDUs"):
        read_ishell_raw(path)


def test_read_ishell_raw_rejects_wrong_detector_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "wrong_shape.fits"
    image = np.zeros((20, 10), dtype=np.int16)
    fits.HDUList(
        [fits.PrimaryHDU(data=image), fits.ImageHDU(data=image), fits.ImageHDU(data=image)]
    ).writeto(path)

    with pytest.raises(RawIShellFitsError, match=r"shape \(20, 10\).+\(2048, 2048\)"):
        read_ishell_raw(path)


@pytest.mark.parametrize(
    ("keyword", "invalid_value", "message"),
    [
        ("ITIME", None, "Missing required ITIME"),
        ("DIVISOR", 0.0, "Invalid DIVISOR"),
        ("CO_ADDS", 1.5, "Invalid CO_ADDS"),
        ("TCS_RA", "not-an-angle", "Invalid TCS_RA"),
    ],
)
def test_read_ishell_raw_rejects_missing_or_invalid_required_metadata(
    native_mef: Path,
    tmp_path: Path,
    keyword: str,
    invalid_value: object,
    message: str,
) -> None:
    path = tmp_path / f"invalid_{keyword.lower()}.fits"
    shutil.copyfile(native_mef, path)
    with fits.open(path, mode="update", memmap=False) as hdus:
        if invalid_value is None:
            del hdus[0].header[keyword]
        else:
            hdus[0].header[keyword] = invalid_value

    with pytest.raises(RawIShellFitsError, match=message):
        read_ishell_raw(path)


def test_returned_arrays_and_header_are_independent_of_closed_fits_file(
    native_mef: Path, tmp_path: Path
) -> None:
    path = tmp_path / "independent.fits"
    shutil.copyfile(native_mef, path)
    frame = read_ishell_raw(path)
    original_values = tuple(
        array[0, 0]
        for array in (
            frame.primary_difference_image,
            frame.pedestal_image,
            frame.signal_image,
            frame.raw_image,
            frame.variance,
            frame.mask,
        )
    )

    with fits.open(path, mode="update", memmap=False) as hdus:
        for hdu in hdus:
            hdu.data[0, 0] = -999
        hdus[0].header["CUSTOM"] = "changed on disk"

    assert (
        tuple(
            array[0, 0]
            for array in (
                frame.primary_difference_image,
                frame.pedestal_image,
                frame.signal_image,
                frame.raw_image,
                frame.variance,
                frame.mask,
            )
        )
        == original_values
    )
    assert frame.header["CUSTOM"] == "preserve me"
    assert all(
        array.flags.owndata
        for array in (
            frame.primary_difference_image,
            frame.pedestal_image,
            frame.signal_image,
            frame.raw_image,
            frame.variance,
            frame.mask,
        )
    )
