from astropy.io import fits

from shuck.calibration.flat import load_flat_info
from shuck.calibration.wavecal import load_wavecal_info, read_line_list
from shuck.resources import REQUIRED_SPEXTOOL_ASSETS, spextool_root
from shuck.telluric import load_ip_coefficients, load_vega_model


def test_bundled_j3_kgas_and_detector_resources_are_locatable(monkeypatch) -> None:
    monkeypatch.delenv("SPEXTOOL5_DIR", raising=False)

    root = spextool_root()

    assert (root / "data" / "version.dat").read_text(encoding="utf-8").strip() == "5.0.3"
    assert all((root / relative).is_file() for relative in REQUIRED_SPEXTOOL_ASSETS)
    for detector_name in (
        "ishell_bias.fits",
        "ishell_lincorr_CDS.fits",
        "ishell_bdpxmk.fits",
        "ishell_htpxmk.fits",
    ):
        header = fits.getheader(root / "instruments" / "ishell" / "data" / detector_name)
        assert header["NAXIS1"] == 2048
        assert header["NAXIS2"] == 2048

    for mode in ("J3", "Kgas"):
        flat = load_flat_info(root, mode)
        wavecal = load_wavecal_info(root, mode)
        lines = read_line_list(wavecal.line_list_path)
        assert flat.mode == mode
        assert wavecal.mode == mode
        assert flat.orders.size > 0
        assert wavecal.orders.size > 0
        assert lines


def test_bundled_telluric_resources_load_without_environment(monkeypatch) -> None:
    monkeypatch.delenv("SPEXTOOL5_DIR", raising=False)
    root = spextool_root()

    vega = load_vega_model(root)
    coefficients = load_ip_coefficients(root, 0.75)

    assert vega.wavelength_micron.size == 1_354_026
    assert coefficients.shape == (3,)
