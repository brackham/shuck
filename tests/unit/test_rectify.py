import numpy as np

from shuck.calibration.rectify import RectificationGeometry, rectify_order


def test_rectify_order_bilinearly_samples_image_and_variance() -> None:
    y, x = np.mgrid[:8, :10]
    image = 2.0 * x + 3.0 * y
    variance = 4.0 + 0.1 * x
    mask = np.zeros_like(image, dtype=np.uint16)
    x_index = np.array([[2.25, 3.25, 4.25], [2.5, 3.5, 4.5]])
    y_index = np.array([[1.5, 1.5, 1.5], [3.5, 3.5, 3.5]])
    geometry = RectificationGeometry(
        order=400,
        reference_pixel=np.array([2.0, 3.0, 4.0]),
        wavelength_micron=np.array([1.2, 1.21, 1.22]),
        spatial_arcsec=np.array([0.0, 1.0]),
        x_index=x_index,
        y_index=y_index,
    )

    rectified = rectify_order(image, variance, mask, geometry)

    np.testing.assert_allclose(rectified.image, 2 * x_index + 3 * y_index)
    np.testing.assert_allclose(rectified.variance, 4 + 0.1 * x_index)
    assert rectified.image.shape == (2, 3)
    assert not np.any(rectified.mask)


def test_rectify_order_flags_samples_outside_detector() -> None:
    geometry = RectificationGeometry(
        order=400,
        reference_pixel=np.array([0.0]),
        wavelength_micron=np.array([1.2]),
        spatial_arcsec=np.array([0.0]),
        x_index=np.array([[-1.0]]),
        y_index=np.array([[0.0]]),
    )

    rectified = rectify_order(
        np.ones((3, 3)),
        np.ones((3, 3)),
        np.zeros((3, 3), dtype=np.uint16),
        geometry,
    )

    assert np.isnan(rectified.image[0, 0])
    assert rectified.mask[0, 0] & (1 << 15)
