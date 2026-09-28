#!/usr/bin/env python3
"""Compare equal-area ringed, spherical, and oblate transits."""

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

import ephesos


RingOrientation = Literal["face_on", "horizontal", "vertical"]
TIME_HOURS = np.linspace(-5.0, 5.0, 241)  # [hour]
PERIOD_DAYS = 4.0  # [day]
EQUIVALENT_RADIUS_RATIO = 0.09
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.12
COSINE_INCLINATION = 0.03
RING_CONFIGURATIONS = {
    "face_on": (
        "face_on_rings",
        "Face-on rings",
    ),
    "horizontal": (
        "horizontal_rings",
        "Horizontally projected rings",
    ),
    "vertical": (
        "vertical_rings",
        "Vertically projected rings",
    ),
}


def evaluate_occultor_models(
    orientation: RingOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate equal-area ringed, spherical, and oblate occultors."""

    if orientation not in RING_CONFIGURATIONS:
        raise ValueError(f"orientation must be one of {tuple(RING_CONFIGURATIONS)}")
    model_arguments = {
        "period_days": PERIOD_DAYS,
        "equivalent_radius_ratio": EQUIVALENT_RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        "cosine_inclination": COSINE_INCLINATION,
    }
    ringed_flux = ephesos.evaluate_projected_occultor_model(
        TIME_HOURS / 24.0,
        occultor_type=RING_CONFIGURATIONS[orientation][0],
        **model_arguments,
    )
    spherical_flux = ephesos.evaluate_projected_occultor_model(
        TIME_HOURS / 24.0,
        occultor_type="disk",
        **model_arguments,
    )
    oblate_flux = ephesos.evaluate_projected_occultor_model(
        TIME_HOURS / 24.0,
        occultor_type="oblate",
        **model_arguments,
    )
    return TIME_HOURS, ringed_flux, spherical_flux, oblate_flux


def run_example(
    output_path: Path,
    orientation: RingOrientation = "vertical",
) -> tuple[np.ndarray, np.ndarray]:
    """Animate one ring orientation against its ringless baseline."""

    time_hours, ringed_flux, spherical_flux, oblate_flux = evaluate_occultor_models(orientation)
    occultor_type, orientation_label = RING_CONFIGURATIONS[orientation]
    ephesos.save_light_curve_animation(
        time_hours,
        ringed_flux,
        output_path,
        title=orientation_label,
        period=PERIOD_DAYS * 24.0,  # [hour]
        radius_ratio=EQUIVALENT_RADIUS_RATIO,
        summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        cosine_inclination=COSINE_INCLINATION,
        occultor_type=occultor_type,
        comparison_models={
            "Spherical planet": spherical_flux,
            "Oblate planet": oblate_flux,
        },
        model_label="Ringed occultor",
        time_label="Time from mid-transit [hour]",
    )
    return time_hours, ringed_flux


def run_orientation_examples(output_directory: Path) -> list[Path]:
    """Write animations for face-on, horizontal, and vertical rings."""

    output_paths = []
    for orientation in RING_CONFIGURATIONS:
        filename = (
            "arbitrary_occultor.gif"
            if orientation == "vertical"
            else f"arbitrary_occultor_{orientation}.gif"
        )
        output_path = output_directory / filename
        run_example(output_path, orientation)
        output_paths.append(output_path)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args()
    run_orientation_examples(arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
