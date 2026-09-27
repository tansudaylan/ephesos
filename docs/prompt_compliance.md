# Prompt compliance

This record maps each applicable requirement in `docs/prompt.md` to the maintained
Ephesos surface.

## Code

- Reusable transit evaluation lives in `ephesos.models.evaluate_transit_model`.
- Reusable figure and animation output lives in `ephesos.visualization`.
- Domain-specific model code remains in Ephesos because light-curve forward
	modeling is its defined ecosystem role.
- Public helpers are concise, typed, documented, and validated with focused tests.
- Variables with physical units use unit-bearing names and unit comments where
	numerical values are assigned.
- Every maintained file read and write reports `Reading from XYZ...` or
	`Writing to XYZ...`.
- GIF assembly uses Pillow through Python paths. It does not invoke shell deletion or ImageMagick.

## Figures

- Each output shows one relative-flux light curve and one comparison to unocculted flux.
- Time labels include units. Relative flux is dimensionless.
- White is the default background. `typeplotback="dark"` selects white axes on a dark background.
- `typefileplot` accepts `png` and `pdf`. PNG output uses 300 dots per inch.
- Figure saving uses tight bounds and a page-width 7.2-inch canvas.
- Gridlines are disabled.
- Titles, labels, tick labels, and legends use one font size.
- Legends are rounded and opaque and identify every plotted line or marker.

## Writing

- README and package documentation use concise, quantitative statements.
- The minimal example states its deterministic assumptions and does not present
	synthetic output as observed evidence.
- Documentation avoids unsupported claims, raw web links, internal drafting notes,
	and semicolon-separated sentences.

## Inapplicable sections

The research-publication, research-proposal, bibliography, and lecture-note
requirements do not apply to this software repository.

## Legacy scope

The monolithic `ephesos.main` engine retains historical formatting debt. Behavioral
changes there are limited to tested plotting defaults, tight frame bounds, and safe
GIF assembly. This avoids a high-risk mechanical rewrite of scientific code.