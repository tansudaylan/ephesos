"""Command-line helpers for Ephesos example workflows."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


def run_output_example(
    run_example: Callable[[Path], Any],
    default_output_path: Path,
    description: str | None,
) -> int:
    """Run an example with a configurable output path."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", type=Path, default=default_output_path)
    arguments = parser.parse_args()
    run_example(arguments.output)
    return 0