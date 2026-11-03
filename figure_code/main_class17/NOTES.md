# Class 17 - Bias-Variance and Double Descent: figure code

Scripts for every code-generated figure loaded by `Class17_BiasVariance_DoubleDescent/A_Textbook_Figures/main.tex` or `Class17_BiasVariance_DoubleDescent/B_Recreated_Figures/main.tex`. Each `<name>.py` is a port of the module in `Class17_BiasVariance_DoubleDescent/B_Recreated_Figures/src/figures/` that wrote `figs/<name>.pdf` (path in each header), with the same data, parameters, seeds and style.

## Verification

Every script was run in a fresh process (`python3 <name>.py`) from an empty `output/`; no caches are used. Each `output/<name>.pdf` and the deck's `B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png` and compared pixel by pixel (mean absolute difference on the 0-255 scale; share of pixels whose largest channel difference exceeds 32).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `error_sources` | `error_sources.py` | B: "Where the error comes from" | 0.000 | 0.000% | match |
| `capacity_fits` | `capacity_fits.py` | B: "Bias-variance Trade-off" | 0.000 | 0.000% | match |
| `tradeoff_curve` | `tradeoff_curve.py` | B: "The trade-off, measured" | 0.000 | 0.000% | match |
| `double_descent` | `double_descent.py` | B: "What is actually measured" | 0.000 | 0.000% | match |
| `double_descent_regimes` | `double_descent_regimes.py` | B: "The same curve, in the ResNet layout" | 0.000 | 0.000% | match |
| `double_descent_noise` | `double_descent_noise.py` | B: "The same shape at four noise levels" | 0.000 | 0.000% | match |
| `minnorm_theory` | `minnorm_theory.py` | B: "Theory against measurement" | 0.000 | 0.000% | match |
| `smooth_interpolants` | `smooth_interpolants.py` | B: "Smoothness and Extra Capacity" | 0.000 | 0.000% | match |
| `sample_wise` | `sample_wise.py` | B: "More data can make a model worse" | 0.000 | 0.000% | match |

9 of 9 match; every figure is pixel-identical at 60 dpi (mean difference 0.000). Slowest script: double_descent_noise (19 s); all 9 together about 53 s. Reruns are byte-identical (`CreationDate` is suppressed).

## Style fallback

`sci_style` is not installed here; `common.py` applies the plain fallback from CONVENTIONS.md (`axes.grid: False`, white figure and axes faces). This deck's figures were built with exactly that fallback: with it every figure matches pixel for pixel, whereas adding the SciencePlots `science` preset makes every figure about 6 px smaller with inward ticks (mean differences 12-30). If `sci_style` is ever installed, the output will differ from the slides for that reason.

## Helper modules

- `common.py` - style, palette, fonts, `save()` / `--out`
- `models.py` - truth function, sampling, piecewise fits, bias-variance and min-norm risk helpers, copied unchanged from `src/common/models.py`
- `double_descent_regimes.py` and `double_descent_noise.py` import `_run`, `PS`, `N` (and `SIGMA`) from `double_descent.py`, exactly as `fig09`/`fig10` imported them from `fig04`. Each still runs on its own.

## Figures with no script

| figure | deck | slide(s) | why |
|---|---|---|---|
| `PerfNoiseBiasVariance` | A | "Where the error comes from" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop4_7a` | A | "Two ways to spend capacity" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop4_7c` | A | "Two ways to spend capacity" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop4_7e` | A | "Two ways to spend capacity" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop4_8` | A | "The trade-off, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PerfBiasVarianceTradeoff` | A | "The trade-off, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_9` | A | "What is actually measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PerfDoubleDescent` | A | "The same shape across four data sets" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PerfSmoothness` | A | "Smoothness is what extra capacity buys" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_10` | A | "Capacity is not the parameter count" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_11` | A | "More data can make a model worse" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_10` | B | "Capacity is not the parameter count" | book crop (`A_Textbook_Figures/figs_official/`) |

The B-variant copies of book figures in `B_Recreated_Figures/figs/` are byte-identical to the files in `A_Textbook_Figures/figs_official/`; no code in the repository writes them. 

TikZ drawn inside the `.tex` is not loaded with `\includegraphics` and needs no script.

No source found: none. Every code-like file in `figs/` that a deck loads is written by a module in `src/`.

## Moved to `figure_code/unused/main_class17/`

None: every module in `src/figures/` writes at least one figure a deck loads.
