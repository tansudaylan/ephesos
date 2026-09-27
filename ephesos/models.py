"""Concise public wrappers around the Ephesos forward model."""

import numpy as np

from .main import eval_modl


def evaluate_transit_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    cosine_inclination: float = 0.0,
    system_type: str = "PlanetarySystem",
    limb_darkening_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one deterministic transit and return its relative flux."""

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < 2 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if radius_ratio <= 0.0:
        raise ValueError("radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    result = eval_modl(
        time_days,
        system_type,
        pericomp=np.array([period_days]),  # [day]
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([summed_radius_to_semimajor_axis]),
        cosicomp=np.array([cosine_inclination]),
        rratcomp=np.array([radius_ratio]),
        coeflmdk=limb_darkening_coefficients,
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    relative_flux = result["rflx"][:, 0]
    return relative_flux
