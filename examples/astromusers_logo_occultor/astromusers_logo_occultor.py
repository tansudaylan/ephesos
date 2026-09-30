#!/usr/bin/env python3
"""Animate the AstroMusers logo transiting against its circular core."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import ephesos


LOGO_PATH = Path(__file__).with_name("AstroMusers.png")
TIME_HOURS = np.linspace(-6.0, 6.0, 241)  # [hour]
PERIOD_DAYS = 4.0  # [day]
CIRCLE_RADIUS_RATIO = 0.22
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.18
COSINE_INCLINATION = 0.02
LOGO_CENTER_PIXELS = (1480.0, 1124.0)
LOGO_CIRCLE_RADIUS_PIXELS = 1124.0


def load_logo_mask(
    logo_path: Path = LOGO_PATH,
    mask_width: int = 301,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return the logo silhouette in units of its central-circle radius."""

    if mask_width < 3:
        raise ValueError("mask_width must be at least three")
    print(f"Reading from {logo_path}...")
    with Image.open(logo_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    foreground = np.min(rgb, axis=2) < 245
    mask_height = round(mask_width * foreground.shape[0] / foreground.shape[1])
    mask_image = Image.fromarray(np.uint8(foreground) * 255)
    resized_mask = mask_image.resize((mask_width, mask_height), Image.Resampling.LANCZOS)
    logo_mask = np.flipud(np.asarray(resized_mask) > 64)

    center_x, center_y = LOGO_CENTER_PIXELS
    radius = LOGO_CIRCLE_RADIUS_PIXELS
    height, width = foreground.shape
    extent = (
        -center_x / radius,
        (width - 1 - center_x) / radius,
        -(height - 1 - center_y) / radius,
        center_y / radius,
    )
    return logo_mask, extent


def evaluate_occultor_models() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Evaluate the complete logo and its circular-core benchmark."""

    logo_mask, logo_extent = load_logo_mask()
    model_arguments = {
        "period_days": PERIOD_DAYS,
        "equivalent_radius_ratio": CIRCLE_RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        "cosine_inclination": COSINE_INCLINATION,
        "grid_size": 501,
    }
    logo_flux = ephesos.evaluate_projected_occultor_model(
        TIME_HOURS / 24.0,
        occultor_type="custom",
        custom_occultor_mask=logo_mask,
        custom_occultor_extent=logo_extent,
        **model_arguments,
    )
    circle_flux = ephesos.evaluate_projected_occultor_model(
        TIME_HOURS / 24.0,
        occultor_type="disk",
        **model_arguments,
    )
    return TIME_HOURS, logo_flux, circle_flux, logo_mask, logo_extent


def run_example(
    output_path: Path,
    max_frames: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    """Animate the logo transit and its circular-core benchmark."""

    time_hours, logo_flux, circle_flux, logo_mask, logo_extent = evaluate_occultor_models()
    ephesos.save_light_curve_animation(
        time_hours,
        logo_flux,
        output_path,
        title="AstroMusers logo transit",
        period=PERIOD_DAYS * 24.0,  # [hour]
        radius_ratio=CIRCLE_RADIUS_RATIO,
        summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        cosine_inclination=COSINE_INCLINATION,
        occultor_type="custom",
        custom_occultor_mask=logo_mask,
        custom_occultor_extent=logo_extent,
        comparison_models={"Circular logo core": circle_flux},
        model_label="Complete AstroMusers logo",
        time_label="Time from mid-transit [hour]",
        max_frames=max_frames,
    )
    return time_hours, logo_flux


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("astromusers_logo_occultor.gif"),
    )
    arguments = parser.parse_args()
    run_example(arguments.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())