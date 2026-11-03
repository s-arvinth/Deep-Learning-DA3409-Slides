# Class 9 — figure scripts

Deck folder: `Class09_MLP_Universal_Approximation_Efficiency` (A_Textbook_Figures and B_Recreated_Figures).

Each `<name>.py` is ported from `B_Recreated_Figures/src/figures/` with the same data, parameters and seeds; shared style, palette, data and models are in `common.py`. `python <name>.py` writes `output/<name>.pdf`; `--out DIR` writes there instead.

## Verification

Every script was run in a fresh process from an empty `output/`, then both PDFs were rendered with `pdftoppm -r 60 -png` and compared (mean absolute pixel difference on 0-255; share of pixels differing by more than 32 in any channel).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| curse_grid (vs B figs/) | curse_grid.py | B: "The curse of dimensionality, drawn" | 0.000 | 0.000% | match |
| curse_volume (vs B figs/) | curse_volume.py | B: "Two more faces of the same problem" | 0.000 | 0.000% | match |
| uat_rate (vs B figs/) | uat_rate.py | A: "The rate, measured"; B: "The rate, measured" | 0.000 | 0.000% | match |
| uat_rate (vs A figs_official/) | uat_rate.py | A: "The rate, measured"; B: "The rate, measured" | 16.213 | 10.067% | MISMATCH — A's figs_official copy is an older render (pre tick-thinning, same data); script matches the current B figure, which the code in src/ produces |
| barron_vs_curse (vs B figs/) | barron_vs_curse.py | A: "The two bounds side by side"; B: "The two bounds side by side" | 0.000 | 0.000% | match |
| barron_vs_curse (vs A figs_official/) | barron_vs_curse.py | A: "The two bounds side by side"; B: "The two bounds side by side" | 11.029 | 7.580% | MISMATCH — A's figs_official copy is an older render (pre tick-thinning, same data); script matches the current B figure, which the code in src/ produces |

## Figures with no script

- Bishop6_3a, Bishop6_3b, Bishop6_3c, Bishop6_4, Bishop6_5, Bishop6_9, Bishop6_10a-d, DeepKLayer — book crops in A figs_official/ (Bishop, Prince)
- TikZ: none of the `\includegraphics` figures in either deck is TikZ.
- No source found: none.

## Original modules copied to `figure_code/unused/main_class9/`

- fig05_depth_width.py (writes depth_width)
- fig06_regions_bounds.py (writes regions_bounds)
- fig07_vc_growth.py (writes vc_growth)

## Other notes

- `depth_width`, `regions_bounds`, `vc_growth` (and `curse_grid` in A figs_official/) exist as PDFs here but no Class 9 deck loads them; the same figures are loaded by Class 10 and scripted in `main_class10/`.

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
