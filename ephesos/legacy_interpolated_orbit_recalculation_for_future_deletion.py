"""Retired interpolated-orbit recalculation retained for future deletion.

The production branch containing this code was deliberately disabled in commit
8747c78 and remained unreachable until it was isolated here. Production modules
must not import this archival implementation.
"""

from collections.abc import Callable
from typing import Any


def recalculate_interpolated_orbits_for_future_deletion(
    model_state: Any,
    calculate_position: Callable,
    summarize: Callable,
) -> None:
    """Preserve the retired global-grid orbit recalculation for review."""

    for time_index in model_state.indxtime:
        for companion_index in model_state.indxcomp:
            xpos, ypos, zpos, _, _, true_anomaly = calculate_position(
                model_state,
                companion_index,
                time_index,
                model_state.phascomp[companion_index][time_index],
            )
            model_state.xposcompgridstar[companion_index] = xpos
            model_state.yposcompgridstar[companion_index] = ypos
            model_state.zposcompgridstar[companion_index] = zpos
            model_state.dictvarborbt["posicompgridprim"][time_index, companion_index] = (
                xpos,
                ypos,
                zpos,
            )
            model_state.dictvarborbt["anomtrue"][time_index, companion_index] = true_anomaly

    if not model_state.booldiag:
        return

    axis_names = ("x", "y", "z")
    for companion_index in model_state.indxcomp:
        for axis_index, axis_name in enumerate(axis_names):
            positions = model_state.dictvarborbt["posicompgridprim"][:, companion_index, axis_index]
            covers_full_orbit = model_state.pericomp[companion_index] < (
                model_state.time[-1] - model_state.time[0]
            )
            is_one_sided = not (positions > 0).any() or not (positions < 0).any()
            if covers_full_orbit and is_one_sided and not (positions == 0).all():
                summarize(model_state.dictvarborbt["anomtrue"][:, companion_index])
                summarize(positions)
                raise ValueError(f"All values are one-sided for {axis_name}-axis")

        if (
            model_state.typecoor == "star"
            and model_state.xposcompgridstar[companion_index].size == 0
        ):
            summarize(model_state.xposcompgridstar[companion_index])
            raise ValueError("xposcompgridstar is empty")
