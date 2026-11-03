# Class 15 - Normalization: figure code

Scripts for every code-generated figure loaded by `Class15_Normalization/A_Textbook_Figures/main.tex` or `Class15_Normalization/B_Recreated_Figures/main.tex`. Each `<name>.py` is a port of the module in `Class15_Normalization/B_Recreated_Figures/src/figures/` that wrote `figs/<name>.pdf` (path in each header), with the same data, parameters, seeds and style.

## Verification

Every script was run in a fresh process (`python3 <name>.py`) from an empty `output/`; no caches are used. Each `output/<name>.pdf` and the deck's `B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png` and compared pixel by pixel (mean absolute difference on the 0-255 scale; share of pixels whose largest channel difference exceeds 32).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `condition_from_data` | `condition_from_data.py` | B: "Two ways the data alone can wreck the conditioning" | 0.000 | 0.000% | match |
| `input_scaling` | `input_scaling.py` | B: "What that does to the error surface" | 0.000 | 0.000% | match |
| `init_rules` | `init_rules.py` | B: "The three rules, and what they do at depth" | 0.000 | 0.000% | match |
| `symmetry` | `symmetry.py` | B: "Symmetry, drawn" | 0.000 | 0.000% | match |
| `drift` | `drift.py` | B: "The statistics do not stay where they were put" | 0.000 | 0.000% | match |
| `bn_effect` | `bn_effect.py` | B: "What it buys" | 0.000 | 0.000% | match |
| `smoothness` | `smoothness.py` | B: "Smoothness, measured" | 0.000 | 0.000% | match |
| `bn_batchsize` | `bn_batchsize.py` | B: "The batch size the statistics depend on" | 0.000 | 0.000% | match |

8 of 8 match; every figure is pixel-identical at 60 dpi (mean difference 0.000). Slowest script: drift (4 s); all 8 together about 18 s. Reruns are byte-identical (`CreationDate` is suppressed).

## Style fallback

`sci_style` is not installed here. The plain fallback in CONVENTIONS.md (`axes.grid`, `figure.facecolor`, `axes.facecolor`) does NOT reproduce this deck's figures: with it every figure came out about 6 px larger at 60 dpi, with outward ticks and plain-text tick labels (mean differences 12-19, MISMATCH throughout). The PDFs in `figs/` were rendered with the SciencePlots `science` preset active: inward ticks on all four sides, minor ticks, 0.05 in save padding, 1.0 default line width, and tick labels typeset as mathtext (`axes.formatter.use_mathtext`; visible in the PDFs as glyph-by-glyph text runs). `common.py` therefore applies the conventions' block plus those rcParams when `import sci_style` fails. With that fallback every figure matches exactly; the font/usetex keys of the preset are omitted because `use()` overrides them.

## Helper modules

- `common.py` - style, palette, fonts, `save()` / `--out`
- `models.py` - data generators, initialisation variances, the MLP, BN/LN and the training loop, copied unchanged from `src/common/models.py`

## Figures with no script

| figure | deck | slide(s) | why |
|---|---|---|---|
| `Bishop7_7` | A | "What normalizing the inputs does" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Train2Exploding` | A | "Fifty layers, five initializations" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop7_8a` | A | "Which entries share a mean" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop7_8b` | A | "Which entries share a mean" | book crop (`A_Textbook_Figures/figs_official/`) |

TikZ drawn inside the `.tex` is not loaded with `\includegraphics` and needs no script.

No source found: none. Every code-like file in `figs/` that a deck loads is written by a module in `src/`.

## Moved to `figure_code/unused/main_class15/`

Copied (not moved; `src/` is untouched) because they write no figure that any deck loads:

- `fig03_variance_depth.py` (writes `variance_depth.pdf`)
- `fig07_norm_axes.py` (writes `norm_axes.pdf`)
- `fig11_scale_invariance.py` (writes `scale_invariance.pdf`)
