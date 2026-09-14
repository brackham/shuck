from pathlib import Path

import numpy as np
from astropy.io import fits

import shuck.telluric as telluric
from shuck.combine import CombinedObservationMetadata, CombinedSpectrum
from shuck.extraction.optimal import ExtractedOrder


def _order(number: int, wavelength: np.ndarray, flux: float, uncertainty: float) -> ExtractedOrder:
    return ExtractedOrder(
        order=number,
        wavelength_micron=wavelength,
        flux=np.full(wavelength.size, flux),
        uncertainty=np.full(wavelength.size, uncertainty),
        mask=np.zeros(wavelength.size, dtype=np.uint16),
        trace_arcsec=np.zeros(wavelength.size),
        background=np.zeros(wavelength.size),
    )


def _combined(order: ExtractedOrder, name: str) -> CombinedSpectrum:
    return CombinedSpectrum(
        orders=(order,),
        scale_order=order.order,
        scale_factors=np.ones(1),
        input_files=(Path(name),),
        sigma_clip=8.0,
        metadata=CombinedObservationMetadata(
            mode="J3",
            mean_mjd=57833.0,
            ra="00:00:00",
            dec="+00:00:00",
            mean_airmass=1.1,
            slit_width_arcsec=0.75,
            plate_scale_arcsec_per_pixel=0.125,
        ),
    )


def test_earth_lsr_velocity_matches_spextool_documented_example() -> None:
    # mc_earthvelocity.pro documents -0.041916911 km/s for this case. The
    # modern Astropy ephemeris differs from its legacy IDL ephemeris by
    # roughly 0.0006 km/s.
    result = telluric.earth_lsr_velocity_kms(57833.0, "00:00:00", "+00:00:00")

    np.testing.assert_allclose(result, -0.041916911, atol=1e-3)


def test_instrument_profile_is_normalized_and_symmetric() -> None:
    profile = telluric.instrument_profile(
        np.arange(-20.0, 21.0), np.array([0.0, 2.7027089, 1.3422317])
    )

    np.testing.assert_allclose(np.sum(profile), 1.0)
    np.testing.assert_allclose(profile, profile[::-1])


def test_build_and_apply_constant_vega_correction(monkeypatch) -> None:
    vega_wavelength = np.linspace(0.9, 1.2, 30_001)
    model = telluric.VegaModel(
        wavelength_micron=vega_wavelength,
        flux=np.full(vega_wavelength.size, 100.0),
        line_continuum=np.full(vega_wavelength.size, 100.0),
        broad_continuum=np.full(vega_wavelength.size, 100.0),
        source_path=Path("vega.sav"),
    )
    monkeypatch.setattr(telluric, "load_vega_model", lambda _: model)
    monkeypatch.setattr(
        telluric,
        "load_ip_coefficients",
        lambda *_: np.array([0.0, 2.7027089, 1.3422317]),
    )
    monkeypatch.setattr(telluric, "earth_lsr_velocity_kms", lambda *_: 0.0)
    wavelength = np.linspace(1.0, 1.1, 101)
    standard = _combined(_order(402, wavelength, 10.0, 0.5), "standard.fits")

    correction = telluric.build_telluric_correction(
        standard,
        standard_group="STANDARD_J3",
        b_magnitude=0.03,
        v_magnitude=0.03,
        radial_velocity_kms=0.0,
        spextool_directory="unused",
    )
    np.testing.assert_allclose(correction.orders[0].flux, 10.0)
    np.testing.assert_allclose(correction.orders[0].uncertainty, 0.5)
    assert correction.standard_orders == standard.orders
    np.testing.assert_allclose(correction.ip_coefficients, [0.0, 2.7027089, 1.3422317])
    assert correction.vega_model_path == Path("vega.sav")

    science = _combined(_order(402, wavelength, 2.0, 0.1), "science.fits")
    corrected = telluric.apply_telluric_correction(science, correction, science_group="SCIENCE_J3")
    np.testing.assert_allclose(corrected.orders[0].flux, 20.0)
    np.testing.assert_allclose(corrected.orders[0].uncertainty, np.sqrt(2.0))


def test_telluric_fits_products_retain_units_and_provenance(tmp_path: Path, monkeypatch) -> None:
    wavelength = np.linspace(1.0, 1.1, 101)
    model = telluric.VegaModel(
        wavelength_micron=np.linspace(0.9, 1.2, 30_001),
        flux=np.full(30_001, 100.0),
        line_continuum=np.full(30_001, 100.0),
        broad_continuum=np.full(30_001, 100.0),
        source_path=Path("vega.sav"),
    )
    monkeypatch.setattr(telluric, "load_vega_model", lambda _: model)
    monkeypatch.setattr(
        telluric,
        "load_ip_coefficients",
        lambda *_: np.array([0.0, 2.7027089, 1.3422317]),
    )
    monkeypatch.setattr(telluric, "earth_lsr_velocity_kms", lambda *_: 0.0)
    standard = _combined(_order(402, wavelength, 10.0, 0.5), "standard.fits")
    science = _combined(_order(402, wavelength, 2.0, 0.1), "science.fits")
    correction = telluric.build_telluric_correction(
        standard,
        standard_group="STANDARD_J3",
        b_magnitude=0.03,
        v_magnitude=0.03,
        radial_velocity_kms=0.0,
        spextool_directory="unused",
    )
    corrected = telluric.apply_telluric_correction(science, correction, science_group="SCIENCE_J3")

    correction_path = telluric.write_telluric_correction(correction, tmp_path / "telluric.fits")
    corrected_path = telluric.write_corrected_spectrum(corrected, tmp_path / "corrected.fits")

    with fits.open(correction_path, checksum=True) as hdus:
        assert hdus[0].header["STAGE"] == "TELLURIC"
        assert hdus[0].header["SLTW_ARC"] == 0.75
        assert hdus[0].header["IPCOEF1"] == 2.7027089
        assert hdus[1].header["TUNIT2"] == "erg cm-2 Angstrom-1 DN-1"
        assert hdus[2].header["TUNIT2"] == "erg s-1 cm-2 Angstrom-1"
    with fits.open(corrected_path, checksum=True) as hdus:
        assert hdus[0].header["STAGE"] == "CORRECTED"
        assert hdus[0].header["OBSMODE"] == "J3"
        assert hdus[0].header["STDAMASS"] == 1.1
        assert hdus[1].header["TUNIT2"] == "erg s-1 cm-2 Angstrom-1"
