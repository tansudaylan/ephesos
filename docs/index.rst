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

.. code-block:: bash

    python examples/minimal_transit.py --typefileplot png
    python examples/arbitrary_occultor.py
    python examples/compact_multiplanet.py
    python examples/run_WhiteDwarf.py
    python examples/run_WASP43.py
    python examples/run_population.py




