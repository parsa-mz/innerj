# The one generated figure, and its specification

Figure 1 of the paper (`paper/images/overview.png`, outside this repository) is a schematic of the two competing accounts. It carries **no data**: it is rendered by an image model from the written specification below, and `fig_overview()` in `innerj/figures/build.py` is kept both as the reproducible fallback and as the specification the render was made from. The reproducibility appendix says so.

**Every other figure in the paper is built by `innerj figures` from an artifact on disk.** Anything carrying a measured number is generated from data and must never be drawn by hand or by a model — a garbled label in a data figure is a wrong result, not a cosmetic defect.

The repository banner (`assets/banner.png`) is also model-generated. It is decoration, appears nowhere in the paper, and carries no claim.

## Raster policy

**Generated images ship as PNG, never converted to PDF.** A raster image wrapped in a PDF gains nothing, hides its true resolution from `pdfinfo`, and makes the alpha-channel black-box failure harder to spot.

Before placing a generated image, check three things:

| check | why | how |
|---|---|---|
| no alpha channel | pdflatex renders RGBA as a black box in some viewers | `Image.open(p).mode` must not contain `A` |
| >= 300 dpi at 6.5in | text width is 6.5in, so width in px / 6.5 | 2000px wide is ~308 dpi |
| smallest label >= 7pt | anything less is unreadable in print | measure the glyph band, `72 * px / dpi` |

Crop the white margin to the content bounding box first; generators leave 2–4% of dead border that costs you resolution at a fixed text width.

**Image models garble small text and subscripts.** Check every label and every symbol character by character; if notation comes back wrong, keep the matplotlib version.

## Shared palette

Used by the schematic and by every data figure, assigned by meaning:

| role | hex |
|---|---|
| residual stream / the focal object | `#277da1` |
| attention | `#f8961e` |
| MLP, and the rejected "gate" | `#9e0059` |
| a fourth ordered level | `#43aa8b` |
| controls, reference lines, quiet text | `#8e8e93` |
| ink | `#1d1d1f` |

## Figure 1 — admission versus transport

Shipped as `paper/images/overview.png`, 2172px = 334 dpi at text width. Verbatim, as used:

```text
Create a publication-quality two-panel conceptual diagram for an academic
machine-learning paper, in a clean minimalist vector style: white background,
generous negative space, precise alignment, modern sans-serif type, thin consistent
strokes, rounded rectangles, no shadows, no gradients, no 3D, no decoration.

Both panels share the same layout so only one difference has to be absorbed: a thin
vertical arrow on the far left labelled "depth" pointing up; a wide rounded rectangle
labelled "passage" containing a faint regular grid of small blue dots at every depth;
and to its right a narrow tall rounded rectangle labelled "query".

Panel a, titled "admission: a gate decides".
Inside the narrow "query" column, place seven solid blue dots evenly spaced from bottom
to top. Draw one horizontal magenta bar across the query column at about 40% height,
labelled "gate" to its right. To the right of the diagram, in quiet grey text:
"z is at the query position at every depth; what varies is whether it is admitted".

Panel b, titled "transport: attention gathers".
The "query" column is empty except in a horizontal band at 40-60% height, which is
tinted pale orange. Three horizontal orange arrows cross from the passage rectangle
into that band, each ending in a solid blue dot inside the query column. Label the band
"gather (attention)" in orange to the right. Lower down, at about 22% height, draw one
dashed grey arrow from the passage toward the query column that terminates in a grey X
instead of a dot, labelled in grey to the right: "installed below, then re-derived from
the passage above". Above the band, draw one vertical blue arrow rising inside the query
column, labelled in blue to the right: "consolidated stream -> answer".

Colours, exactly: blue #277da1, orange #f8961e, magenta #9e0059, grey #8e8e93,
near-black #1d1d1f for titles.

All explanatory prose sits in the right margin. Nothing overlaps the diagram. Aspect
ratio about 3:1, readable at 6.5 inches wide.
```

**Two things a schematic of this mechanism must not imply**, both corrected on 2026-08-03 and worth re-checking against any future render:

- Not "MLPs never transport". Inside the window no tested MLP output contributes positively, but `mlp.L15` on language and `mlp.L42` on tracking are both significantly positive *outside* it. Scope any MLP label to the gather layer.
- Not that the site generalises. The gather is at L39 on the language family and L48 on tracking. Either label the layer as "L39 (language family)" or leave it unlabelled.
