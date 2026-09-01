"""Observation and constitutive operators for support-aware inversion."""

from .support import (
    SupportOperator,
    apply_surface_crossing,
    box_volume_operator,
    build_error_covariance,
    build_observation_operator,
    gaussian_kernel_operator,
    interval_operator,
    normalized_misfit,
    point_trilinear_operator,
    sample_profile_correlated_noise,
)

__all__ = [
    "SupportOperator",
    "apply_surface_crossing",
    "box_volume_operator",
    "build_error_covariance",
    "build_observation_operator",
    "gaussian_kernel_operator",
    "interval_operator",
    "normalized_misfit",
    "point_trilinear_operator",
    "sample_profile_correlated_noise",
]
