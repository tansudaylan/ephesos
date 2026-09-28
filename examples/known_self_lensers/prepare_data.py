#!/usr/bin/env python3
"""Prepare compact Kepler self-lensing light curves from public archives."""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from astropy.io import fits
from astroquery.mast import Observations
import numpy as np


OUTPUT_DIRECTORY = Path(__file__).with_name("data")
KOI3278_SOURCE = (
    "https://raw.githubusercontent.com/ethankruse/koi3278/master/"
    "KOI3278_events_sap.txt"
)


def center_times(time_days: np.ndarray, epoch_days: float, period_days: float) -> np.ndarray:
    """Return event-centered times wrapped to half an orbital period."""

    return (time_days - epoch_days + 0.5 * period_days) % period_days - 0.5 * period_days


def detrend_event(
    time_days: np.ndarray,
    relative_flux: np.ndarray,
    relative_flux_error: np.ndarray,
    exclusion_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a quadratic local baseline measured outside the pulse."""

    baseline = np.abs(time_days) > exclusion_days
    coefficients = np.polyfit(time_days[baseline], relative_flux[baseline], 2)
    continuum = np.polyval(coefficients, time_days)
    return relative_flux / continuum, relative_flux_error / continuum


def bin_light_curve(
    time_days: np.ndarray,
    relative_flux: np.ndarray,
    relative_flux_error: np.ndarray,
    bin_width_hours: float,
) -> np.ndarray:
    """Return inverse-variance weighted points on a fixed event-centered grid."""

    time_hours = 24.0 * time_days  # [hour]
    edges = np.arange(
        np.floor(time_hours.min() / bin_width_hours) * bin_width_hours,
        time_hours.max() + 2.0 * bin_width_hours,
        bin_width_hours,
    )
    bin_index = np.digitize(time_hours, edges) - 1
    rows = []
    for index in range(edges.size - 1):
        selected = bin_index == index
        if not np.any(selected):
            continue
        weights = relative_flux_error[selected] ** -2
        weighted_flux = np.average(relative_flux[selected], weights=weights)
        formal_error = 1.0 / np.sqrt(np.sum(weights))
        rows.append((np.average(time_hours[selected], weights=weights), weighted_flux, formal_error))
    return np.asarray(rows)


def prepare_koi3278(temporary_directory: Path) -> None:
    """Stack the public author-reduced SAP pulse windows for KOI-3278."""

    source_path = temporary_directory / "KOI3278_events_sap.txt"
    print(f"Reading from {KOI3278_SOURCE}...")
    urlretrieve(KOI3278_SOURCE, source_path)
    print(f"Reading from {source_path}...")
    time_days, relative_flux, relative_flux_error = np.loadtxt(source_path, unpack=True)
    finite = np.isfinite(relative_flux_error)
    time_days = time_days[finite]
    relative_flux = relative_flux[finite]
    relative_flux_error = relative_flux_error[finite]

    period_days = 88.1805979  # [day]
    epoch_days = 85.4189422  # [BJD - 2455000]
    centered_days = center_times(time_days, epoch_days, period_days)
    event_number = np.rint((time_days - epoch_days) / period_days).astype(int)
    detrended_flux = np.empty_like(relative_flux)
    detrended_error = np.empty_like(relative_flux_error)
    for number in np.unique(event_number):
        selected = event_number == number
        detrended_flux[selected], detrended_error[selected] = detrend_event(
            centered_days[selected],
            relative_flux[selected],
            relative_flux_error[selected],
            exclusion_days=0.28,  # [day]
        )

    selected = np.abs(centered_days) < 0.55  # [day]
    table = bin_light_curve(
        centered_days[selected],
        detrended_flux[selected],
        detrended_error[selected],
        bin_width_hours=0.75,  # [hour]
    )
    output_path = OUTPUT_DIRECTORY / "koi3278_kepler.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=(
            "time_from_pulse_hours,relative_flux,relative_flux_error\n"
            "Source: Kruse & Agol (2014), DOI 10.1126/science.1251999; "
            "author-reduced Kepler SAP events from github.com/ethankruse/koi3278.\n"
            "Processing: finite-error pulse windows, per-event quadratic baseline outside "
            "|dt|=0.28 day, inverse-variance bins of width 0.75 hour."
        ),
        comments="# ",
        fmt=("%.6f", "%.9f", "%.9f"),
    )


def prepare_kic8145411(temporary_directory: Path) -> None:
    """Extract and stack quality-filtered PDC pulse windows for KIC 8145411."""

    observations = Observations.query_criteria(
        obs_collection="Kepler",
        target_name="kplr008145411",
    )
    products = Observations.get_product_list(observations)
    products = Observations.filter_products(
        products,
        productSubGroupDescription="LLC",
    )
    manifest = Observations.download_products(
        products,
        download_dir=str(temporary_directory),
        cache=True,
    )

    time_blocks = []
    flux_blocks = []
    error_blocks = []
    for file_path in manifest["Local Path"]:
        path = Path(file_path)
        print(f"Reading from {path}...")
        with fits.open(path, memmap=False) as hdus:
            data = hdus[1].data
            quality = data["SAP_QUALITY"] == 0
            finite = (
                np.isfinite(data["TIME"])
                & np.isfinite(data["PDCSAP_FLUX"])
                & np.isfinite(data["PDCSAP_FLUX_ERR"])
            )
            selected = quality & finite
            median_flux = np.median(data["PDCSAP_FLUX"][selected])
            time_blocks.append(np.asarray(data["TIME"][selected], dtype=float))
            flux_blocks.append(np.asarray(data["PDCSAP_FLUX"][selected] / median_flux, dtype=float))
            error_blocks.append(
                np.asarray(data["PDCSAP_FLUX_ERR"][selected] / median_flux, dtype=float)
            )

    time_days = np.concatenate(time_blocks)
    relative_flux = np.concatenate(flux_blocks)
    relative_flux_error = np.concatenate(error_blocks)
    period_days = 455.826  # [day]
    epoch_days = 267.867  # [BKJD]
    centered_days = center_times(time_days, epoch_days, period_days)
    event_number = np.rint((time_days - epoch_days) / period_days).astype(int)
    event_window = np.abs(centered_days) < 1.2  # [day]

    detrended_flux = np.empty(np.sum(event_window))
    detrended_error = np.empty(np.sum(event_window))
    selected_time = centered_days[event_window]
    selected_event = event_number[event_window]
    source_flux = relative_flux[event_window]
    source_error = relative_flux_error[event_window]
    for number in np.unique(selected_event):
        selected = selected_event == number
        detrended_flux[selected], detrended_error[selected] = detrend_event(
            selected_time[selected],
            source_flux[selected],
            source_error[selected],
            exclusion_days=0.42,  # [day]
        )

    central = np.abs(selected_time) < 0.65  # [day]
    table = bin_light_curve(
        selected_time[central],
        detrended_flux[central],
        detrended_error[central],
        bin_width_hours=1.0,  # [hour]
    )
    output_path = OUTPUT_DIRECTORY / "kic8145411_kepler.csv"
    print(f"Writing to {output_path}...")
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=(
            "time_from_pulse_hours,relative_flux,relative_flux_error\n"
            "Source: Kepler long-cadence PDC light curves from MAST; system parameters "
            "from Masuda et al. (2019), arXiv:1907.07656.\n"
            "Processing: SAP_QUALITY=0, finite PDC values, per-event quadratic baseline "
            "outside |dt|=0.42 day, inverse-variance bins of width 1 hour."
        ),
        comments="# ",
        fmt=("%.6f", "%.9f", "%.9f"),
    )


def main() -> int:
    with TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory)
        prepare_koi3278(path)
        prepare_kic8145411(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())