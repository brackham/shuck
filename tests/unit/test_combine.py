from pathlib import Path

import numpy as np

from shuck.combine import combine_exposures, determine_scale_factors
from shuck.extraction.optimal import ExtractedExposure, ExtractedOrder


def test_determine_scale_factors_matches_median_reference() -> None:
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.5, 1.0, 1.5]])

    scales = determine_scale_factors(spectra, np.zeros_like(spectra, dtype=np.uint16))

    np.testing.assert_allclose(scales, [1.0, 0.5, 2.0])


def _exposure(name: str, scale: float, outlier: bool = False) -> ExtractedExposure:
    wavelength = np.linspace(1.3, 1.31, 20)
    true_flux = np.linspace(10, 20, 20)
    flux = true_flux / scale
    if outlier:
        flux = flux.copy()
        flux[10] = 1e6
    order = ExtractedOrder(
        order=402,
        wavelength_micron=wavelength,
        flux=flux,
        uncertainty=np.full(20, 0.2 / scale),
        mask=np.zeros(20, dtype=np.uint16),
        trace_arcsec=np.zeros(20),
        background=np.zeros(20),
    )
    return ExtractedExposure(
        orders=(order,),
        profiles=(),
        apertures=(),
        traces=(),
        source_path=Path(name),
    )


def test_combine_exposures_scales_all_spectra_and_rejects_outlier() -> None:
    exposures = (
        _exposure("a.fits", 1.0),
        _exposure("b.fits", 0.5),
        _exposure("c.fits", 2.0, outlier=True),
    )

    product = combine_exposures(exposures, scale_order=402, sigma_clip=8.0)

    expected = np.linspace(10, 20, 20)
    np.testing.assert_allclose(product.scale_factors, [1.0, 0.5, 2.0])
    np.testing.assert_allclose(product.orders[0].flux, expected)
    assert product.orders[0].uncertainty[10] > product.orders[0].uncertainty[9]
