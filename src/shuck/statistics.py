"""SpeXTool-compatible robust combination helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustWeightedMean:
    """Result of a robust inverse-variance weighted mean."""

    mean: np.ndarray
    variance: np.ndarray
    good: np.ndarray


def robust_weighted_mean(
    data: np.ndarray,
    variance: np.ndarray,
    *,
    axis: int = 0,
    sigma: float = 8.0,
    mask: np.ndarray | None = None,
) -> RobustWeightedMean:
    """Return SpeXTool's sigma-clipped inverse-variance weighted mean.

    The behavioral references are SpeXTool 5.0.3 ``mc_meancomb.pro`` and
    ``mc_findoutliers.pro``. Outliers are samples farther than ``sigma``
    scaled MADs from the even median, where scaled MAD is
    ``1.482 * median(abs(x - median(x)))``. The returned variance is
    ``1 / sum(1 / variance)`` over accepted samples.

    Unlike IDL's historical implementation, invalid and non-positive
    variances are always excluded explicitly so they cannot produce a
    plausible but undefined weighted result.
    """

    values = np.asarray(data, dtype=np.float64)
    variances = np.asarray(variance, dtype=np.float64)
    if values.shape != variances.shape:
        raise ValueError("data and variance must have identical shapes")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    if axis < -values.ndim or axis >= values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array of dimension {values.ndim}")
    axis %= values.ndim

    good = np.isfinite(values) & np.isfinite(variances) & (variances > 0)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape != values.shape:
            raise ValueError("mask must have the same shape as data")
        good &= supplied

    masked_values = np.where(good, values, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(masked_values, axis=axis, keepdims=True)
        mad = 1.482 * np.nanmedian(
            np.abs(masked_values - median),
            axis=axis,
            keepdims=True,
        )
        deviation = np.abs(values - median)
        reject = deviation / mad > sigma

    # With zero MAD, SpeXTool retains values identical to the median and
    # rejects any finite non-identical values (0/0 is not greater than sigma).
    zero_mad = mad == 0
    reject = np.where(zero_mad, deviation > 0, reject)
    good &= ~reject

    weights = np.zeros_like(variances)
    np.divide(1.0, variances, out=weights, where=good)
    weight_sum = np.sum(weights, axis=axis, dtype=np.float64)
    weighted_values = np.zeros_like(values)
    np.multiply(values, weights, out=weighted_values, where=good)
    numerator = np.sum(weighted_values, axis=axis, dtype=np.float64)
    mean = np.full(weight_sum.shape, np.nan, dtype=np.float64)
    mean_variance = np.full(weight_sum.shape, np.inf, dtype=np.float64)
    np.divide(numerator, weight_sum, out=mean, where=weight_sum > 0)
    np.divide(1.0, weight_sum, out=mean_variance, where=weight_sum > 0)
    return RobustWeightedMean(mean=mean, variance=mean_variance, good=good)
