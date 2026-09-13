from pathlib import Path

import numpy as np

from shuck.calibration.flat import (
    FlatInfo,
    _smooth_fiterpolate,
    find_order_edges,
    find_vertical_order_offset,
    rotate_to_processing,
)


def _flat_info(shape: tuple[int, int] = (80, 100)) -> FlatInfo:
    mask = np.zeros(shape, dtype=np.int16)
    # Stored native mask becomes rows 20:31, columns 10:91 after ROTATION=5.
    processing_mask = np.zeros(shape, dtype=np.int16)
    processing_mask[20:31, 10:91] = 402
    mask[:] = np.fliplr(processing_mask)
    coefficients = np.array([[[20.0, 0.0], [30.0, 0.0]]])
    return FlatInfo(
        mode="J3",
        rotation=5,
        slit_height_arcsec=5.0,
        slit_height_pixels=10.0,
        slit_height_range=(7, 14),
        orders=np.array([402]),
        plate_scale_arcsec_per_pixel=0.5,
        resolving_power_pixels=1.0,
        step=5,
        edge_fraction=0.85,
        com_window=5,
        edge_degree=1,
        norm_nxgrid=4,
        norm_nygrid=3,
        oversample=1.0,
        ybuffer=1,
        ycor_order=402,
        xranges=np.array([[10, 90]], dtype=np.int32),
        edge_coefficients=coefficients,
        order_mask_native=mask,
        source_path=Path("J3_flatinfo.fits"),
    )


def test_rotation_five_is_left_right_flip_in_numpy_coordinates() -> None:
    image = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(rotate_to_processing(image, 5), np.fliplr(image))


def test_fiterpolate_surface_reproduces_quadratic_input() -> None:
    y, x = np.indices((31, 51), dtype=np.float64)
    image = 10 + 2 * x + 3 * y + 0.2 * x**2 - 0.1 * y**2 + 0.05 * x * y

    smooth = _smooth_fiterpolate(image, 5, 3)

    np.testing.assert_allclose(smooth, image, rtol=1e-11, atol=1e-9)


def test_vertical_order_offset_tracks_shifted_flat() -> None:
    info = _flat_info()
    flat = np.zeros(info.order_mask_native.shape)
    flat[23:34, 10:91] = 100.0

    offset = find_vertical_order_offset(flat, info)

    assert offset == -3


def test_find_order_edges_recovers_straight_synthetic_order() -> None:
    info = _flat_info()
    flat = np.zeros(info.order_mask_native.shape)
    flat[20:31, 10:91] = 100.0

    coefficients, xranges = find_order_edges(flat, info, vertical_offset=0)

    x = np.array([20.0, 50.0, 80.0])
    bottom = np.polynomial.polynomial.polyval(x, coefficients[0, 0])
    top = np.polynomial.polynomial.polyval(x, coefficients[0, 1])
    np.testing.assert_allclose(bottom, 19.5, atol=1.0)
    np.testing.assert_allclose(top, 30.5, atol=1.0)
    np.testing.assert_array_equal(xranges, [[10, 90]])
