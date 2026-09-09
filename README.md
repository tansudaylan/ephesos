# Ephesos

## Scientific purpose
Ephesos is a light-curve forward-modeling framework for planetary and stellar systems. It computes the relative flux signal from a geometric and brightness model of a system and is intended for transparent model evaluation and diagnostic visualization.

The core scientific workflow is built around evaluating the sky-projected brightness distribution of bodies in a system via the `ephesos.eval_modl()` entry point, enabling investigations of transits, star spots, occultations, phase curves, eclipse mapping, microlensing, and reflected light.

## Repository role in the ecosystem
Ephesos is a scientific modeling library within the wider astrophysical analysis stack. It complements the time-domain and cataloging workflows by making the forward model explicit and inspectable, so researchers can trace how physical assumptions move into the final light curve.

## Installation

```bash
cd /path/to/ephesos
pip install -e .
```

## Minimal usage

```python
import ephesos

# The primary forward-modeling entry point is exposed through the package.
# Example usage depends on the target system configuration and is typically
# constructed through the model setup functions in the package.
```

## Model diagnostics
A useful forward-model run should make the following visible:

- the input system geometry and stellar properties;
- the intermediate brightness model or limb-darkening assumptions;
- the resulting relative flux light curve;
- any residuals or model comparison diagnostics.

This makes the scientific assumptions inspectable without having to reverse-engineer the source code.

## Current maintenance status
Ephesos remains a research-grade scientific library rather than a broad generic astronomy toolkit. The supported package surface is the importable model and evaluation functions, while legacy analysis scripts and notebooks are kept as historical or exploratory material unless they are explicitly migrated into the active API.

