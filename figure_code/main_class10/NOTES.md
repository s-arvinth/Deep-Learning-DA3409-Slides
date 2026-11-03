# Class 10 — figure scripts

Deck folder: `Class10_Depth_Width_Efficiency_Theorems` (A_Textbook_Figures and B_Recreated_Figures).

Each `<name>.py` is ported from `B_Recreated_Figures/src/figures/` with the same data, parameters and seeds; shared style, palette, data and models are in `common.py`. `python <name>.py` writes `output/<name>.pdf`; `--out DIR` writes there instead.

## Verification

Every script was run in a fresh process from an empty `output/`, then both PDFs were rendered with `pdftoppm -r 60 -png` and compared (mean absolute pixel difference on 0-255; share of pixels differing by more than 32 in any channel).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| depth_width (vs B figs/) | depth_width.py | A: "The separation, measured"; B: "The separation, measured" | 0.000 | 0.000% | match |
| depth_width (vs A figs_official/) | depth_width.py | A: "The separation, measured"; B: "The separation, measured" | 0.000 | 0.000% | match |
| compositional (vs B figs/) | compositional.py | A: "The size of the effect"; B: "The size of the effect" | 0.000 | 0.000% | match |
| compositional (vs A figs_official/) | compositional.py | A: "The size of the effect"; B: "The size of the effect" | 0.000 | 0.000% | match |
| depth_width_asymmetry (vs B figs/) | depth_width_asymmetry.py | A: "The cost of flattening"; B: "The cost of flattening" | 0.000 | 0.000% | match |
| depth_width_asymmetry (vs A figs_official/) | depth_width_asymmetry.py | A: "The cost of flattening"; B: "The cost of flattening" | 0.000 | 0.000% | match |
| regions_bounds (vs B figs/) | regions_bounds.py | B: "Regions against parameters" | 0.000 | 0.000% | match |
| vc_growth (vs B figs/) | vc_growth.py | A: "VC dimension, drawn"; B: "VC dimension, drawn" | 0.000 | 0.000% | match |
| vc_growth (vs A figs_official/) | vc_growth.py | A: "VC dimension, drawn"; B: "VC dimension, drawn" | 0.000 | 0.000% | match |

## Figures with no script

- DeepParams — book crop in A figs_official/ (Prince)
- TikZ: none of the `\includegraphics` figures in either deck is TikZ.
- No source found: none.

## Original modules copied to `figure_code/unused/main_class10/`

- none (every module in src/figures/ writes a figure one of the decks loads)

## Other notes

- Class 10 A loads `compositional`, `depth_width`, `depth_width_asymmetry`, `vc_growth` from its figs_official/; those files are byte-identical to B figs/, so the same scripts cover both variants.

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
