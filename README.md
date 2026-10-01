# Ephesos

## Purpose
Ephesos is a light-curve forward-modeling framework for planetary and stellar
systems. It computes relative flux from explicit geometry and brightness models.

The core workflow evaluates sky-projected brightness through `ephesos.eval_modl()`.
The concise `ephesos.evaluate_transit_model()` interface covers deterministic
single-companion transit calculations.

Ephesos also predicts transit times. `evaluate_linear_transit_times(epochs, epoch_time,
orbital_period)` gives a linear ephemeris; `evaluate_sinusoidal_ttv(epochs, offset,
phase, amplitude, ttv_period)` gives a timing residual with the same time units as
its offset and amplitude. `evaluate_nbody_transit_times(planet_parameters,
stellar_mass_solar, start_time_days, step_days, step_count)` uses `ttvfast` and
returns each planet's predicted epoch indices and times in days. Each row of
`planet_parameters` contains mass, period, eccentricity, inclination, longitude
of ascending node, argument of periapsis, and mean anomaly in `ttvfast`'s units.
Missing transit events are excluded without renumbering subsequent epochs. The
sinusoidal residual is descriptive rather than a dynamical interaction model;
the N-body calculation requires the optional `ttvfast` integrator.

## Light-curve modeling
Ephesos predicts light curves from explicit orbital geometry, companion
sizes, surface-brightness profiles, and limb-darkening assumptions.

## Installation

```bash
cd /path/to/ephesos
pip install -e .
export EPHESOS_PATH=/path/to/ephesos
```

For N-body transit-time predictions, install `pip install -e '.[ttv]'`.

`EPHESOS_PATH` identifies the repository root. Runtime inputs belong under `data/`.
Generated pipeline outputs belong under `visuals/`. Git ignores both directories.

## Minimal usage

The runnable example evaluates a central transit with a three-day orbital period,
a companion-to-star radius ratio of 0.1, a summed-radius-to-semimajor-axis ratio
of 0.1, and quadratic limb darkening.

```bash
python examples/minimal_transit/minimal_transit.py --typefileplot png
```

![Ephesos deterministic transit-model prediction](examples/minimal_transit/minimal_transit.png)

The curve is a deterministic prediction under these assumptions. It contains no
observed or randomly generated data. The model produces a 1.13% transit depth.

Single-system examples write compact GIFs without external data or
command-line animation tools. The population example writes a static corner plot
of features derived from the synthesized light curves.

```bash
python examples/minimal_transit/minimal_transit.py
python examples/PlanetsWithDisks/PlanetsWithDisks.py
python examples/arbitrary_occultor/arbitrary_occultor.py
python examples/astromusers_logo_occultor/astromusers_logo_occultor.py
python examples/compact_multiplanet/compact_multiplanet.py
python examples/emitting_companion/emitting_companion.py
python examples/known_self_lensers/known_self_lensers.py
python examples/run_WhiteDwarf/run_WhiteDwarf.py
python examples/run_WASP43/run_WASP43.py
python examples/run_population/run_population.py
python examples/self_lensing/self_lensing.py
python examples/transits_simultaneous/transits_simultaneous.py
```

## Notebook examples

Five executable notebooks provide interactive entry points to complementary
capabilities:

- [`examples/capabilities/capabilities.ipynb`](examples/capabilities/capabilities.ipynb)
	inspects the installed API, evaluates transit, self-lensing, and projected-ring
	models, checks failure behavior, and benchmarks representative calls using
	deterministic simulated systems.
- [`examples/known_self_lensers/known_self_lensers.ipynb`](examples/known_self_lensers/known_self_lensers.ipynb)
	compares bundled public Kepler observations with Ephesos self-lensing models.
- [`examples/PlanetsWithDisks/PlanetsWithDisks.ipynb`](examples/PlanetsWithDisks/PlanetsWithDisks.ipynb)
	compares deterministic equal-area spherical and circumplanetary-disk silhouettes.
- [`examples/emitting_companion/emitting_companion.ipynb`](examples/emitting_companion/emitting_companion.ipynb)
	compares planetary emission levels and their secondary-eclipse depths.
- [`examples/run_WASP43/run_WASP43.ipynb`](examples/run_WASP43/run_WASP43.ipynb)
	displays a deterministic WASP-43 b transit prediction as an animation.

Run each notebook from its own example directory. Figures are written directly
to that directory's `visuals/` folder.

![Transit, projected-ring, and self-lensing capability comparison](examples/capabilities/visuals/capability_comparison.png)

![Known Kepler self-lensers from the notebook pipeline](examples/known_self_lensers/visuals/known_self_lensers.png)

![Equal-area circumplanetary-disk morphologies from the notebook pipeline](examples/PlanetsWithDisks/visuals/circumplanetary_disk_morphologies.png)

![Secondary eclipses from emitting planetary companions](examples/emitting_companion/visuals/emitting_companion.png)

![WASP-43 b transit-model animation](examples/run_WASP43/run_WASP43.gif)

Single-system animations are written beside their scripts. The population summary
is written to `examples/run_population/population_features.png` and shows transit
depth, total duration, ingress duration, and equivalent width for 2,048 deterministic
oblate and ringed systems. It also shows each feature's departure from a matched
spherical planet with the same projected occulting area and orbit.

![Corner plot of derived transiting-planet population features](examples/run_population/population_features.png)

The ringed-occultor workflow generates face-on, horizontally projected, and
vertically projected ring examples. Each animation compares the ringed model
with spherical and oblate planets on the same orbit. Every shape has the same
projected occulting area, isolating light-curve differences caused by shape.

![Equal-area face-on ring, spherical planet, and oblate planet transits](examples/arbitrary_occultor/arbitrary_occultor_face_on.gif)

![Equal-area horizontal ring, spherical planet, and oblate planet transits](examples/arbitrary_occultor/arbitrary_occultor_horizontal.gif)

![Equal-area vertical ring, spherical planet, and oblate planet transits](examples/arbitrary_occultor/arbitrary_occultor.gif)

The AstroMusers example uses the complete group logo as the transiter and compares
its light curve with the logo's central circular component on the same orbit.

![AstroMusers logo transit compared with its circular core](examples/astromusers_logo_occultor/astromusers_logo_occultor.gif)

The self-lensing example compares deterministic 30-day binaries containing
$0.6\,M_\odot$ white-dwarf, $1.4\,M_\odot$ neutron-star, and $8\,M_\odot$
black-hole companions orbiting a solar analog. A controlled white-dwarf
comparison isolates the effects of quadratic limb darkening and projected
impact parameter. These are explicit toy systems rather than fits to observed
data.

![Toy self-lensing systems and finite-source effects](examples/self_lensing/self_lensing.png)

The known-self-lenser example compares Ephesos predictions with public Kepler
observations of KOI-3278 and KIC 8145411. The bundled compact tables contain
author-reduced simple-aperture photometry from Kruse and Agol (2014) for
KOI-3278 and quality-filtered Pre-search Data Conditioning photometry from the
Mikulski Archive for Space Telescopes for KIC 8145411. The latter uses the pulse
ephemeris and system parameters reported by Masuda et al. (2019). The Ephesos
curves use the published masses, radii, periods, and impact parameters without
fitting the plotted data.

The current Ephesos self-lensing interface assumes circular motion and omits
finite-lens occultation, third-light dilution, and Kepler cadence integration.
Its curves are therefore local finite-source approximations rather than
reproductions of the papers' eccentric-orbit fits. The data files retain source
and reduction metadata. Run `prepare_data.py` in the example directory to
reproduce them from the public sources when network access is available.

![Known Kepler self-lensers compared with Ephesos models](examples/known_self_lensers/known_self_lensers.png)

The circumplanetary-disk example compares face-on and edge-on disk silhouettes
with a spherical planet of equal projected occulting area. Its lower panel shows
the shape-induced departure from that spherical benchmark in parts per thousand.

![Equal-area circumplanetary-disk transit comparison](examples/PlanetsWithDisks/PlanetsWithDisks.png)

The simultaneous-transit example places two planets in an illustrative 3:2
resonant pair with a shared mid-transit time. The animation compares their
combined light curve with each planet's isolated contribution.

![Two simultaneous transits compared with isolated-planet models](examples/transits_simultaneous/transits_simultaneous.gif)

The compact-system example synthesizes an illustrative seven-planet
TRAPPIST-1-like resonant chain rather than fitting observations. It produces 21
transit events over 12 days. Its minimum adjacent separation is 6.54 mutual Hill
radii, exceeding the circular coplanar pairwise threshold of 3.46.

The reveal variant preserves the complete green light-curve history.

![Compact seven-planet reveal animation](examples/compact_multiplanet/compact_multiplanet_reveal.gif)

The trailing variant keeps only the latest three transit durations green.
Its playback slows by factors of 2 during full transits, 4 during ingress and
egress, and 6 during simultaneous transits.

![Compact seven-planet trailing-history animation](examples/compact_multiplanet/compact_multiplanet_trailing.gif)

## Reusable plotting

`ephesos.save_light_curve_figure()` writes `png` output at 300 dots per inch or
vector `pdf` output. It uses a white background by default and accepts
`typeplotback="dark"`. `ephesos.save_light_curve_animation()` applies the same
labels, colors, typography, and opaque legend to GIF output. Its default
`light_curve_mode="reveal"` retains the complete green history. The alternative
`light_curve_mode="trailing"` shows a moving recent-history window over the gray
curve. That window defaults to three measured transit durations and can be set
explicitly with `history_duration` in the time-axis units.

## Model diagnostics
Ephesos diagnostics show:

- the input system geometry and stellar properties
- the intermediate brightness model or limb-darkening assumptions
- the resulting relative flux light curve
- any residuals or model comparison diagnostics

Together these products expose the assumptions that determine the modeled light curve.

