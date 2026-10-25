Ephesos
========

I use Ephesos to predict how a specified planetary or stellar system changes
the light we receive. Set the geometry and brightness assumptions, then inspect
the resulting relative flux. Ephesos calculates predictions for supplied
parameters. Downstream packages handle fitting and posterior sampling.

Animated examples show a planetary transit and two simultaneous transits:

.. image:: ../examples/run_WASP43/run_WASP43.gif
    :alt: WASP-43 b crossing its star in the Ephesos transit model
    :width: 720 px

.. image:: ../examples/transits_simultaneous/transits_simultaneous.gif
    :alt: Two planets crossing their star at the same time
    :width: 720 px

.. toctree::
    :maxdepth: 2
    :caption: User guide

    getting_started
    api

Installation
------------

.. code-block:: bash

    pip install -e .
    export EPHESOS_PATH=/path/to/ephesos

Transit evaluation
------------------

.. code-block:: python

    import numpy as np
    import ephesos

    time_days = np.linspace(-0.1, 0.1, 201)  # [day]
    relative_flux = ephesos.evaluate_transit_model(
         time_days,
         period_days=3.0,
         radius_ratio=0.1,
         summed_radius_to_semimajor_axis=0.1,
    )

Visualization
-------------

``save_light_curve_figure`` writes a focused ``png`` or ``pdf`` figure.
``save_light_curve_animation`` writes a compact GIF. Both functions use a white
background by default and accept ``typeplotback="dark"``.

Examples
--------

The emitting-companion example shows how planetary surface brightness sets the
secondary-eclipse depth. The self-lensing example compares planetary, white-dwarf,
neutron-star, and black-hole toy systems and isolates the effects of limb
darkening and projected alignment. The known-self-lenser example overlays
public Kepler observations of KOI-3278 and KIC 8145411 with Ephesos circular
finite-source predictions based on published system parameters. The latter
predictions are approximations because the current interface omits eccentric
motion, finite-lens occultation, dilution, and cadence integration.

.. code-block:: bash

    python examples/minimal_transit/minimal_transit.py --typefileplot png
    python examples/PlanetsWithDisks/PlanetsWithDisks.py
    python examples/PlanetsWithDisks/PlanetsWithDisks.py --bokeh
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

The four-lens example uses a solar-mass, solar-radius source and a 30-day orbit
with impact parameter 0.2. Its Jupiter-mass lens produces a pulse of about
0.8 ppm. The shared Chalcedon calculation averages magnification over numerical
pixels and integrates the central pixel containing the lens singularity.
This avoids suppressing the signal when the Einstein radius is smaller than a
pixel. Figures are saved under ``examples/self_lensing/visuals/`` with timestamps,
and existing outputs are preserved. These are lensing-only calculations with
the finite size of the occulting lens omitted.

Ephesos and SpaceEngine
----------------------

Ephesos calculates relative flux from specified systems and viewing geometry.
Its examples expose the arrays behind the light curves, alongside projected
views and animations. This makes it possible to compare a model with measured
photometry and to test a numerical approximation against an analytic result.

`SpaceEngine <https://spaceengine.org/>`_ provides an interactive, freely moving
observer, catalog and procedurally generated planetary systems, terrain,
atmospheres, and accelerated or reversed orbital motion. Its
`graphics manual <https://spaceengine.org/manual/user-manual/>`_ describes
eclipse shadows, planet shine, volumetric rings, and gravitational-lensing
visualization for compact objects. Those capabilities make it useful for
exploring what a system looks like from different locations.

The same manual also makes clear why a displayed brightness needs care in a
photometric comparison. Antialiasing, screen resolution, dynamic image updates,
and animated dithering affect the rendered pixels. A matched light-curve
comparison would require identical orbital elements, stellar brightness,
observer geometry, time samples, and a stated conversion from rendered images
to integrated flux. Ephesos's limb darkening and finite-source lensing would
need matching settings as well.

This comparison uses the documented SpaceEngine capabilities and executed
Ephesos examples. SpaceEngine's listed operating systems are Windows 10/11,
and a SpaceEngine run was not performed here. A cross-engine flux residual,
accuracy ranking, or rendering-speed comparison therefore remains unmeasured.




