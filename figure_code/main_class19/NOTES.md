# Class 19 - Regularization Heuristics: figure code

Scripts for every code-generated figure loaded by `Class19_Regularization_Heuristics/A_Textbook_Figures/main.tex` or `Class19_Regularization_Heuristics/B_Recreated_Figures/main.tex`. Each `<name>.py` is a port of the module in `Class19_Regularization_Heuristics/B_Recreated_Figures/src/figures/` that wrote `figs/<name>.pdf` (path in each header), with the same data, parameters, seeds and style.

## Verification

Every script was run in a fresh process (`python3 <name>.py`) from an empty `output/`; no caches are used. Each `output/<name>.pdf` and the deck's `B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png` and compared pixel by pixel (mean absolute difference on the 0-255 scale; share of pixels whose largest channel difference exceeds 32).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `learning_curves` | `learning_curves.py` | B: "The learning curve is a measurement, not a prediction" | 0.000 | 0.000% | match |
| `early_stopping_path` | `early_stopping_path.py` | B: "Descent on a quadratic, drawn" | 0.000 | 0.000% | match |
| `early_vs_decay` | `early_vs_decay.py` | B: "The two filters, measured" | 0.000 | 0.000% | match |
| `ensemble_gain` | `ensemble_gain.py` | B: "The promise, and what arrives" | 0.000 | 0.000% | match |
| `dropout` | `dropout.py` | B: "Dropout, measured" | 0.000 | 0.000% | match |
| `input_noise` | `input_noise.py` | B: "The equivalence, checked" | 0.000 | 0.000% | match |
| `bayesian` | `bayesian.py` | B: "The posterior, drawn" | 0.000 | 0.000% | match |

7 of 7 match; every figure is pixel-identical at 60 dpi (mean difference 0.000). Slowest script: dropout (25 s); all 7 together about 35 s. Reruns are byte-identical (`CreationDate` is suppressed).

## Style fallback

`sci_style` is not installed here; `common.py` applies the plain fallback from CONVENTIONS.md (`axes.grid: False`, white figure and axes faces). This deck's figures were built with exactly that fallback: with it every figure matches pixel for pixel, whereas adding the SciencePlots `science` preset makes every figure about 6 px smaller with inward ticks (mean differences 12-30). If `sci_style` is ever installed, the output will differ from the slides for that reason.

## Helper modules

- `common.py` - style, palette, fonts, `save()` / `--out`
- `models.py` - truth, sampling, features, SGD paths, ensembles, the filters and the Bayesian posterior, copied unchanged from `src/common/models.py`
- Split: `fig02_early_vs_decay.py` -> `early_stopping_path.py` + `early_vs_decay.py`. Neither part draws random numbers, so the split cannot change any data.

## Figures with no script

| figure | deck | slide(s) | why |
|---|---|---|---|
| `Bishop9_7a` | A | "The learning curve is a measurement, not a prediction" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_7b` | A | "The learning curve is a measurement, not a prediction" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_8` | A | "The two filters, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegEnsemble` | A | "The promise, and what arrives" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_17` | A | "An ensemble you do not have to train" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegDropout` | A | "Dropout, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegNoise` | A | "The equivalence, checked" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegTransferMultiC` | A | "When the data set is the constraint" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_1` | A | "Augmentation" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegBayesian` | A | "The posterior, drawn" | book crop (`A_Textbook_Figures/figs_official/`) |
| `P_RegTransferMultiC` | B | "The three routes, drawn" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_1` | B | "Augmentation, on an image" | book crop (`A_Textbook_Figures/figs_official/`) |

The B-variant copies of book figures in `B_Recreated_Figures/figs/` are byte-identical to the files in `A_Textbook_Figures/figs_official/`; no code in the repository writes them. 

TikZ drawn inside the `.tex` is not loaded with `\includegraphics` and needs no script.

No source found: none. Every code-like file in `figs/` that a deck loads is written by a module in `src/`.

## Moved to `figure_code/unused/main_class19/`

None: every module in `src/figures/` writes at least one figure a deck loads.
