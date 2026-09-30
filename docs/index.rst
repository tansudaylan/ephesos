Ephesos
========

Purpose
-------

Ephesos evaluates deterministic forward models for planetary and stellar light
curves. The package exposes the model geometry, brightness assumptions, and
relative-flux prediction for inspection and reuse.

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
secondary-eclipse depth. The self-lensing example compares white-dwarf,
neutron-star, and black-hole toy systems and isolates the effects of limb
darkening and projected alignment. The known-self-lenser example overlays
public Kepler observations of KOI-3278 and KIC 8145411 with Ephesos circular
finite-source predictions based on published system parameters. The latter
predictions are approximations because the current interface omits eccentric
motion, finite-lens occultation, dilution, and cadence integration.

.. code-block:: bash

    python examples/minimal_transit/minimal_transit.py --typefileplot png
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




