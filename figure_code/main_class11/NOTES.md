# Class 11 — figure scripts

Deck folder: `Class11_Activations_Errors_Backprop_AD` (A_Textbook_Figures and B_Recreated_Figures).

Each `<name>.py` is ported from `B_Recreated_Figures/src/figures/` with the same data, parameters and seeds; shared style, palette, data and models are in `common.py`. `python <name>.py` writes `output/<name>.pdf`; `--out DIR` writes there instead.

## Verification

Every script was run in a fresh process from an empty `output/`, then both PDFs were rendered with `pdftoppm -r 60 -png` and compared (mean absolute pixel difference on 0-255; share of pixels differing by more than 32 in any channel).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| activation_zoo_a (vs B figs/) | activation_zoo_a.py | B: "The activation zoo: the rectifier family" | 0.000 | 0.000% | match |
| activation_zoo_b (vs B figs/) | activation_zoo_b.py | B: "The activation zoo: smooth and saturating" | 0.000 | 0.000% | match |
| activation_derivatives_a (vs B figs/) | activation_derivatives_a.py | B: "Differentiated: the rectifier family" | 0.000 | 0.000% | match |
| activation_derivatives_b (vs B figs/) | activation_derivatives_b.py | B: "Differentiated: smooth and saturating" | 0.000 | 0.000% | match |
| saturation_bound (vs B figs/) | saturation_bound.py | A: "What backpropagation actually uses"; B: "Saturation, drawn" | 0.000 | 0.000% | match |
| saturation_bound (vs A figs_official/) | saturation_bound.py | A: "What backpropagation actually uses"; B: "Saturation, drawn" | 0.000 | 0.000% | match |
| dying_relu (vs B figs/) | dying_relu.py | A: "The rectifier and its derivative"; B: "A unit switching off, drawn" | 0.000 | 0.000% | match |
| dying_relu (vs A figs_official/) | dying_relu.py | A: "The rectifier and its derivative"; B: "A unit switching off, drawn" | 0.000 | 0.000% | match |
| vanishing_exploding (vs B figs/) | vanishing_exploding.py | B: "Measured, over twenty layers" | 0.000 | 0.000% | match |
| diff_methods (vs B figs/) | diff_methods.py | B: "Why not just perturb each weight?" | 0.000 | 0.000% | match |
| heteroscedastic (vs B figs/) | heteroscedastic.py | B: "When one number per input is not enough" | 0.000 | 0.000% | match |
| mixture_density (vs B figs/) | mixture_density.py | B: "When the target is genuinely multi-valued" | 0.000 | 0.000% | match |

## Figures with no script

- Bishop6_12a-f, Bishop6_17a-b, Bishop6_18, Bishop6_19a-d, Bishop8_2, Bishop8_4, Bishop8_5, LossHeteroscedastic, PrinceCompGraph, PrinceExploding — book crops in A figs_official/ (Bishop, Prince)
- TikZ: none of the `\includegraphics` figures in either deck is TikZ.
- No source found: none.

## Original modules copied to `figure_code/unused/main_class11/`

- fig07_ad_modes.py (writes ad_modes, loaded by neither variant)

## Other notes

- `fig01_activation_zoo.py` and `fig02_activation_derivatives.py` each wrote two figures from one build(); they are split into `activation_zoo_a/_b.py` and `activation_derivatives_a/_b.py`.
- `figs/activation_zoo.pdf`, `figs/activation_derivatives.pdf` (stale, no code writes them) and `figs/ad_modes.pdf` are not loaded by either deck, so they get no script.
- Class 11 A loads `dying_relu` and `saturation_bound` from figs_official/; those are byte-identical to B figs/.

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
