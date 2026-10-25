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
windows must not overlap. Event midpoints and durations come from interpolated
0.1%-depth contacts. Undetected or incomplete events are returned as NaN.
Masked light-curve and ephemeris inputs raise ``ValueError`` before their
hidden values can enter the measurement. Fully unmasked containers are supported.

Rømer's 1676 interpretation of Io's immersion into and emergence from
Jupiter's shadow used changing eclipse times to infer that light travels at a
finite speed. The Ephesos example below illustrates that argument with
synthetic timings, not historical observations. It assumes circular,
coplanar Earth and Jupiter orbits, a fixed eclipse duration, and seeded
30-second timing noise. Time zero is opposition, with Earth between the Sun
and Jupiter and their separation smallest. Orbital values come from NASA NSSDCA's Earth, Jupiter,
and Jovian satellite fact sheets. The historical source is Rømer (1676),
``Démonstration touchant le mouvement de la lumière``, *Journal des sçavans*.

.. code-block:: console

   python examples/romer_light_time/romer_light_time.py --typefileplot png

The two panels show the synthetic observed-minus-ephemeris delays against time
and Earth-Jupiter distance. Fitting the delay slope recovers the injected speed
of light.

.. image:: ../examples/romer_light_time/visuals/romer_light_time.png
   :alt: Synthetic Io eclipse-contact delays versus time and Earth-Jupiter distance with a fitted light-time relation.

Planet transits in an eclipsing binary
--------------------------------------

The next example combines Ephesos's luminous-companion forward model with its
primary-star transit model. It treats the secondary as a uniform-brightness
sphere and places the planet transits between the binary eclipses. Since the
events do not overlap, their flux changes can be added directly. The model uses
independent ephemerides and omits binary reflex motion and three-body dynamics.

.. code-block:: console

   python examples/planet_in_eclipsing_binary/planet_in_eclipsing_binary.py --typefileplot png

The upper panel compares the combined light curve with the binary alone. The
lower panel isolates the primary transits.

.. image:: ../examples/planet_in_eclipsing_binary/visuals/planet_in_eclipsing_binary.png
   :alt: Synthetic binary eclipses and isolated planetary transits across the primary over the full 30-day baseline.
