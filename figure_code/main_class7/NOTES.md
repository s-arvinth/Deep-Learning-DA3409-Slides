# Class 7 — figure scripts

Deck folder: `Class07_Deep_Networks_Universal_Approximation` (A_Textbook_Figures and B_Recreated_Figures).

Each `<name>.py` is ported from `B_Recreated_Figures/src/figures/` with the same data, parameters and seeds; shared style, palette, data and models are in `common.py`. `python <name>.py` writes `output/<name>.pdf`; `--out DIR` writes there instead.

## Verification

Every script was run in a fresh process from an empty `output/`, then both PDFs were rendered with `pdftoppm -r 60 -png` and compared (mean absolute pixel difference on 0-255; share of pixels differing by more than 32 in any channel).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| compose (vs B figs/) | compose.py | B: "Composing the two networks" | 0.000 | 0.000% | match |
| fold (vs B figs/) | fold.py | B: "The same thing, in the plots" | 0.000 | 0.000% | match |
| compose2d (vs B figs/) | compose2d.py | B: "Composing in two dimensions" | 0.000 | 0.000% | match |
| buildup (vs B figs/) | buildup.py | B: "Computed step by step" | 0.000 | 0.000% | match |
| sawtooth (vs B figs/) | sawtooth.py | B: "The sawtooth, drawn" | 0.000 | 0.000% | match |
| ua_ridge (vs B figs/) | ua_ridge.py | B: "The construction, drawn" | 0.000 | 0.000% | match |
| regions (vs B figs/) | regions.py | B: "Linear regions per parameter" | 0.000 | 0.000% | match |
| representation (vs B figs/) | representation.py | B: "Representation learning: the result" | 0.000 | 0.000% | match |
| hierarchy (vs B figs/) | hierarchy.py | B: "A hierarchy, built explicitly" | 0.000 | 0.000% | match |

## Figures with no script

- Bishop6_8 — book figure (Bishop & Bishop 2024, Fig. 6.8) in B figs/; src/README says it is not generated
- Bishop6_11, Bishop6_15, DeepBuildUp, DeepConcat, DeepFold, DeepKLayer, DeepParams, DeepTwoLayer, DeepTwoLayer2D — book crops in A figs_official/ (Bishop, Prince)
- TikZ: none of the `\includegraphics` figures in either deck is TikZ.
- No source found: none.

## Original modules copied to `figure_code/unused/main_class7/`

- none (every module in src/figures/ writes a figure one of the decks loads)

## Style note: the sci_style fallback

`sci_style` is not installed here. With only the fallback listed in CONVENTIONS.md
(`axes.grid`/`figure.facecolor`/`axes.facecolor`) every figure comes out slightly
different from the one in the deck: ticks point outwards and only on the bottom/left,
the bounding box is 0.1 in larger, and tick labels are plain text instead of mathtext
(mean diff 3-8, a few % of pixels over 32). The deck's PDFs were evidently rendered
with the SciencePlots `science` preset active. `common.py` therefore applies, in the
`except ImportError` branch, the CONVENTIONS fallback **plus** the preset values that
the style's own `RC` does not override (inward ticks on all sides, `savefig.pad_inches`
0.05, `lines.linewidth` 1.0, `axes.formatter.use_mathtext`, ...). With that block every
script reproduces its figure pixel-for-pixel; the same block was confirmed by running
the untouched original `src/` modules on a scratch copy, which also only match with it.
