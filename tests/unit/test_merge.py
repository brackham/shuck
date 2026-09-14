from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from shuck.combine import CombinedObservationMetadata
from shuck.extraction.optimal import ExtractedOrder
from shuck.merge import merge_orders, write_merged_spectrum


def _order(
    number: int,
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: float,
    mask: int = 0,
) -> ExtractedOrder:
    return ExtractedOrder(
        order=number,
        wavelength_micron=wavelength,
        flux=flux,
        uncertainty=np.full(wavelength.size, uncertainty),
        mask=np.full(wavelength.size, mask, dtype=np.uint16),
        trace_arcsec=np.zeros(wavelength.size),
        background=np.zeros(wavelength.size),
    )


def test_merge_orders_inverse_variance_averages_overlap_and_preserves_native_edges() -> None:
    first = _order(10, np.array([1.0, 1.1, 1.2, 1.3]), np.full(4, 10.0), 2.0, 1)
    second = _order(11, np.array([0.8, 0.9, 1.0, 1.1]), np.full(4, 20.0), 1.0, 2)

    merged = merge_orders((first, second))

    np.testing.assert_allclose(merged.wavelength_micron, [0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    np.testing.assert_allclose(merged.flux, [20.0, 20.0, 18.0, 18.0, 10.0, 10.0])
    np.testing.assert_allclose(
        merged.uncertainty,
        [1.0, 1.0, np.sqrt(0.8), np.sqrt(0.8), 2.0, 2.0],
    )
    np.testing.assert_array_equal(merged.mask, [2, 2, 3, 3, 1, 1])
    assert merged.overlap_ranges == ((1.0, 1.1),)


def test_merge_orders_accepts_descending_input_and_keeps_internal_nan() -> None:
    first = _order(
        10,
        np.array([1.3, 1.2, 1.1, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0]),
        0.1,
    )
    second = _order(11, np.array([1.15, 1.2, 1.25]), np.full(3, 2.0), 0.1)

    merged = merge_orders((first, second))

    assert np.all(np.diff(merged.wavelength_micron) > 0)
    at_12 = np.flatnonzero(np.isclose(merged.wavelength_micron, 1.2))
    assert at_12.size == 1
    assert merged.flux[at_12[0]] == 2.0


def test_write_merged_spectrum_is_checksummed(tmp_path: Path) -> None:
    order = _order(10, np.array([1.0, 1.1]), np.array([3.0, 4.0]), 0.2)
    product = merge_orders(
        (order,),
        science_group="SCIENCE_J3",
        standard_group="STANDARD_J3",
        science_files=(Path("science.fits"),),
        standard_files=(Path("standard.fits"),),
        science_airmass=1.2,
        standard_airmass=1.1,
        observation_metadata=CombinedObservationMetadata(
            mode="J3",
            mean_mjd=61136.0,
            ra="12:00:00",
            dec="+10:00:00",
            mean_airmass=1.2,
            slit_width_arcsec=0.75,
            plate_scale_arcsec_per_pixel=0.125,
        ),
    )

    path = write_merged_spectrum(product, tmp_path / "merged.fits")

    with fits.open(path, checksum=True) as hdus:
        assert hdus[0].header["STAGE"] == "MERGED"
        assert hdus[0].header["ORDERS"] == "10"
        assert hdus[0].header["SCIGRP"] == "SCIENCE_J3"
        assert hdus[0].header["STDGRP"] == "STANDARD_J3"
        assert hdus[0].header["OBSMODE"] == "J3"
        assert hdus[0].header["AMDIFF"] == pytest.approx(-0.1)
        assert hdus[1].header["TUNIT2"] == "erg s-1 cm-2 Angstrom-1"
        np.testing.assert_allclose(hdus[1].data["FLUX"], [3.0, 4.0])


def test_merge_orders_marks_nonoverlapping_inner_edges_nan_like_spextool() -> None:
    first = _order(10, np.array([1.0, 1.1]), np.array([3.0, 4.0]), 0.2, 1)
    second = _order(11, np.array([1.3, 1.4]), np.array([5.0, 6.0]), 0.3, 2)

    merged = merge_orders((first, second))

    np.testing.assert_allclose(merged.wavelength_micron, [1.0, 1.1, 1.3, 1.4])
    np.testing.assert_allclose(merged.flux[[0, 3]], [3.0, 6.0])
    assert np.all(np.isnan(merged.flux[[1, 2]]))
    np.testing.assert_array_equal(merged.mask, [1, 0, 0, 2])
    assert merged.overlap_ranges == (None,)


def test_merge_orders_uses_anchor_overlap_grid_when_second_order_is_contained() -> None:
    first = _order(10, np.linspace(1.0, 1.5, 6), np.full(6, 10.0), 1.0)
    second = _order(11, np.array([1.15, 1.25, 1.35]), np.full(3, 20.0), 1.0)

    merged = merge_orders((first, second))

    np.testing.assert_allclose(merged.wavelength_micron, [1.0, 1.2, 1.3, 1.4, 1.5])
    assert merged.overlap_ranges == ((1.0, 1.35),)
