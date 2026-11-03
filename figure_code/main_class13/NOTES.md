# Class 13 — figure scripts

Deck folder: `Class13_Gradient_Descent_Optimization` (A_Textbook_Figures and B_Recreated_Figures).

Each `<name>.py` is ported from `B_Recreated_Figures/src/figures/` with the same data, parameters and seeds; shared style, palette, data and models are in `common.py`. `python <name>.py` writes `output/<name>.pdf`; `--out DIR` writes there instead.

## Verification

Every script was run in a fresh process from an empty `output/`, then both PDFs were rendered with `pdftoppm -r 60 -png` and compared (mean absolute pixel difference on 0-255; share of pixels differing by more than 32 in any channel).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| stationary_types (vs B figs/) | stationary_types.py | B: "The three signatures" | 0.000 | 0.000% | match |
| quadratic_ellipses (vs B figs/) | quadratic_ellipses.py | B: "Reading the curvature off the contours" | 0.000 | 0.000% | match |
| steepest_direction (vs B figs/) | steepest_direction.py | B: "The steepest direction, drawn" | 0.000 | 0.000% | match |
| descent_lemma (vs B figs/) | descent_lemma.py | B: "The guarantee, as a function of the step" | 0.000 | 0.000% | match |
| linreg_descent (vs B figs/) | linreg_descent.py | B: "Descent on a convex loss" | 0.000 | 0.000% | match |
| gabor_landscape (vs B figs/) | gabor_landscape.py | B: "A loss that is not convex" | 0.000 | 0.000% | match |
| sgd_escape (vs B figs/) | sgd_escape.py | B: "Noise is the feature" | 0.000 | 0.000% | match |
| batch_variance (vs B figs/) | batch_variance.py | B: "Unbiased, and the price of the square root" | 0.000 | 0.000% | match |
| batch_tradeoff (vs B figs/) | batch_tradeoff.py | B: "What the batch size buys" | 0.000 | 0.000% | match |
| gradient_payoff (vs B figs/) | gradient_payoff.py | B: "The counting argument, drawn" | 0.000 | 0.000% | match |
| shuffling (vs B figs/) | shuffling.py | B: "Why the data has to be shuffled" | 0.000 | 0.000% | match |
| convex_vs_nonconvex (vs B figs/) | convex_vs_nonconvex.py | B: "Convex and non-convex, side by side" | 0.000 | 0.000% | match |
| curvature (vs B figs/) | curvature.py | B: "What $\lambda_{\max}$ actually is" | 0.000 | 0.000% | match |
| step_regimes_2d (vs B figs/) | step_regimes_2d.py | B: "The four regimes on the contour map" | 0.000 | 0.000% | match |
| step_regimes_3d (vs B figs/) | step_regimes_3d.py | B: "The four regimes on the surface" | 0.000 | 0.000% | match |

## Figures with no script

- Bishop7_1 — book figure (Bishop & Bishop 2024, Fig. 7.1) in B figs/; src/README says it is kept as published
- Bishop7_2, TrainConvexProb, TrainGaborData, TrainGaborGDSGD, TrainGaborMin, TrainGaborModel, TrainGaborSGDIter, TrainLRMin, TrainLineSearch — book crops in A figs_official/ (Bishop, Prince)
- TikZ: none of the `\includegraphics` figures in either deck is TikZ.
- No source found: none.

## Original modules copied to `figure_code/unused/main_class13/`

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
