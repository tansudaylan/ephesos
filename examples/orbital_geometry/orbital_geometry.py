#!/usr/bin/env python3
"""Plot orbital geometry through Ephesos's Euler rotations and observer views."""

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import ephesos


ORBITING_BODIES = (
    ephesos.OrbitingBody(
        "Planet b",
        semimajor_axis=18.0,  # [stellar radii]
        eccentricity=0.12,
        argument_of_periapsis_deg=25.0,  # [deg]
        inclination_deg=38.0,  # [deg]
        longitude_of_ascending_node_deg=15.0,  # [deg]
        true_anomaly_deg=65.0,  # [deg]
    ),
    ephesos.OrbitingBody(
        "Planet c",
        semimajor_axis=34.0,  # [stellar radii]
        eccentricity=0.08,
        argument_of_periapsis_deg=130.0,  # [deg]
        inclination_deg=62.0,  # [deg]
        longitude_of_ascending_node_deg=95.0,  # [deg]
        true_anomaly_deg=210.0,  # [deg]
    ),
    ephesos.OrbitingBody(
        "Planet d",
        semimajor_axis=56.0,  # [stellar radii]
        eccentricity=0.04,
        argument_of_periapsis_deg=275.0,  # [deg]
        inclination_deg=24.0,  # [deg]
        longitude_of_ascending_node_deg=155.0,  # [deg]
        true_anomaly_deg=315.0,  # [deg]
    ),
)


def run_example(output_path: Path, typefileplot: str = "png") -> Path:
    """Save a multipanel view of a deterministic three-planet system."""

    return ephesos.save_orbital_geometry_figure(
        ORBITING_BODIES,
        output_path,
        star_radius=1.0,  # [stellar radii]
        length_unit="stellar radii",
        viewing_angles_deg=((35.0, 25.0), (125.0, 35.0)),  # [deg]
        observer_positions=((7.0, -4.0, 3.0),),  # [stellar radii]
        field_of_view_deg=130.0,  # [deg]
        typefileplot=typefileplot,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("orbital_geometry.png"),
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
