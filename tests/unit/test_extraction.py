from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shuck.calibration.rectify import RectifiedOrder
from shuck.extraction.optimal import extract_order_optimal
from shuck.extraction.preprocess import PreprocessedExposure, combine_preprocessed_exposures
from shuck.extraction.profile import SpatialProfile, find_apertures, make_spatial_profile
from shuck.extraction.trace import trace_order


def _synthetic_order() -> tuple[RectifiedOrder, np.ndarray]:
    spatial = np.linspace(0, 5, 41)
    wavelength = np.linspace(1.30, 1.31, 240)
    true_trace = 2.45 + 0.08 * (wavelength - wavelength.mean()) / np.ptp(wavelength)
    flux = 120 + 15 * np.sin(np.linspace(0, 4 * np.pi, len(wavelength)))
    profile = np.exp(-0.5 * ((spatial[:, None] - true_trace[None, :]) / 0.28) ** 2)
    profile /= np.sum(profile, axis=0)
    image = 7.0 + profile * flux[None, :]
    return (
        RectifiedOrder(
            order=402,
            wavelength_micron=wavelength,
            spatial_arcsec=spatial,
            image=image,
            variance=np.full_like(image, 0.04),
            mask=np.zeros_like(image, dtype=np.uint16),
        ),
        flux,
    )


def test_automatic_profile_aperture_and_trace_recover_source() -> None:
    order, _ = _synthetic_order()

    profile = make_spatial_profile(order)
    aperture = find_apertures((profile,))[0]
    trace = trace_order(order, aperture, degree=2)

    assert aperture.valid
    assert aperture.sign == 1
    np.testing.assert_allclose(aperture.position_arcsec, 2.45, atol=0.05)
    assert trace.valid
    assert trace.rms_arcsec < 0.02


def test_optimal_extraction_recovers_synthetic_flux() -> None:
    order, expected_flux = _synthetic_order()
    profile = make_spatial_profile(order)
    aperture = find_apertures((profile,))[0]
    trace = trace_order(order, aperture, degree=2)

    extracted = extract_order_optimal(order, profile, aperture, trace)

    good = (extracted.mask == 0) & np.isfinite(extracted.flux)
    assert np.mean(good) > 0.98
    np.testing.assert_allclose(extracted.flux[good], expected_flux[good], rtol=0.015)
    assert np.all(extracted.uncertainty[good] > 0)


def test_aperture_finding_uses_cross_order_location_for_weak_order() -> None:
    spatial = np.linspace(0, 5, 41)
    source = np.exp(-0.5 * ((spatial - 2.8) / 0.25) ** 2)
    strong = [
        SpatialProfile(order=400 + index, spatial_arcsec=spatial, normalized_flux=source)
        for index in range(4)
    ]
    edge_artifact = np.exp(-0.5 * ((spatial - 0.05) / 0.1) ** 2)
    profiles = tuple(
        strong
        + [
            SpatialProfile(
                order=404,
                spatial_arcsec=spatial,
                normalized_flux=0.05 * source + edge_artifact,
            )
        ]
    )

    locations = find_apertures(profiles)

    assert all(location.valid for location in locations)
    np.testing.assert_allclose([location.position_arcsec for location in locations], 2.8, atol=0.1)


def test_group_image_robustly_rejects_exposure_outlier() -> None:
    order, _ = _synthetic_order()
    exposures = []
    for index, offset in enumerate((0.0, 0.0, 1000.0)):
        image = order.image.copy()
        image[10, 20] += offset
        exposures.append(
            PreprocessedExposure(
                orders=(
                    RectifiedOrder(
                        order=order.order,
                        wavelength_micron=order.wavelength_micron,
                        spatial_arcsec=order.spatial_arcsec,
                        image=image,
                        variance=order.variance,
                        mask=order.mask,
                    ),
                ),
                metadata=SimpleNamespace(),
                source_path=Path(f"{index}.fits"),
                dark_files=(),
                flat_files=(),
            )
        )

    combined = combine_preprocessed_exposures(tuple(exposures))

    np.testing.assert_allclose(combined.orders[0].image, order.image)
