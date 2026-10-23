Getting started
===============

Ephesos evaluates deterministic forward models of transit, occultation,
self-lensing, planetary emission, and transit-timing signals. Models return
predicted quantities for supplied parameters. Downstream analysis packages
handle fitting and posterior sampling.

Installation
------------

Install Ephesos from its repository root:

.. code-block:: console

   python -m pip install -e .

Set ``EPHESOS_PATH`` to the repository root. Runtime inputs are read from
``data/`` and generated examples write products beneath their example directory
or ``visuals/``.

First run
---------

The minimal transit example evaluates a central transit with a three-day period
and a companion-to-star radius ratio of 0.1:

.. code-block:: console

   python examples/minimal_transit/minimal_transit.py --typefileplot png

The resulting curve is a deterministic prediction under the configured orbital
geometry and limb-darkening law. The example uses no observed light curve. Use
:func:`ephesos.models.evaluate_transit_model` to evaluate the same class of
model from Python. The :doc:`api` page lists the other supported model families.

Stellar activity
----------------

Use ``ephesos.stellar_activity`` for stellar-rotation, spot, and flare forward
models. The latitude-dependent period law uses
P(latitude) = P_equator * (1 + shear * sin(latitude)**2), with latitude in
degrees. Here ``shear`` is the fractional pole-to-equator period difference.

Run the native models with the standard installation:

.. code-block:: console

   python examples/differential_rotation/differential_rotation.py --typefileplot png
   python examples/spot_flare/spot_flare.py --typefileplot png

``evaluate_spot_flare_model`` accepts explicit spot and flare arrays, times in
days, and angles in degrees. The native calculation integrates visible
spherical-cap spots on the quadratically limb-darkened disk, including partial
visibility at the limb. The supplied radius ratio is the sine of the cap's
angular radius. Longitude zero places an equatorial spot nearest the observer
at time zero. Gaussian spot evolution is optional. Independent spot deficits
are added without accounting for spot overlap. Increase ``quadrature_order``
from its default of 64 to test convergence. Flare amplitudes are maxima on the
supplied time grid. No optional spot engine is required.
Miletos provides parameter parsing, likelihoods, and fitting. Ephesos provides
the forward models. The examples are synthetic predictions.

Optional N-body timings
-----------------------

Install the ``ttv`` extra to use :func:`ephesos.models.evaluate_nbody_transit_times`:

.. code-block:: console

   python -m pip install -e '.[ttv]'

That function delegates integration to ``ttvfast``. Linear and sinusoidal
ephemerides do not require that extra.

The :doc:`api` includes a two-planet example that plots N-body transit-time
residuals. Run it from the repository root after installing the ``ttv`` extra:

.. code-block:: console

   python examples/photodynamical_ttv/photodynamical_ttv.py

For an observed light curve normalized to unit out-of-transit flux,
``ephesos.models.derive_transit_timing_variations`` estimates per-event timing
and duration changes around supplied linear-ephemeris transit epochs. The
windows must not overlap. Event midpoints are estimated from interpolated
0.1%-depth contacts, and undetected or incomplete events are returned as NaN.
This is a threshold-based measurement, not a transit-profile fit.
