Getting started
===============

Ephesos evaluates deterministic forward models of transit, occultation,
self-lensing, planetary emission, and transit-timing signals. Models return
predicted quantities for supplied parameters; fitting and posterior sampling
are provided by downstream analysis packages.

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
geometry and limb-darkening law. It is not a fit to observations. Use
:func:`ephesos.models.evaluate_transit_model` to evaluate the same class of
model from Python; :doc:`api` lists the other supported model families.

Optional N-body timings
-----------------------

Install the ``ttv`` extra to use :func:`ephesos.models.evaluate_nbody_transit_times`:

.. code-block:: console

   python -m pip install -e '.[ttv]'

That function delegates integration to ``ttvfast``. Linear and sinusoidal
ephemerides do not require that extra.