# Class 18 - Regularization: Explicit and Implicit: figure code

Scripts for every code-generated figure loaded by `Class18_Regularization_Explicit_Implicit/A_Textbook_Figures/main.tex` or `Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/main.tex`. Each `<name>.py` is a port of the module in `Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/src/figures/` that wrote `figs/<name>.pdf` (path in each header), with the same data, parameters, seeds and style.

## Verification

Every script was run in a fresh process (`python3 <name>.py`) from an empty `output/`; no caches are used. Each `output/<name>.pdf` and the deck's `B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png` and compared pixel by pixel (mean absolute difference on the 0-255 scale; share of pixels whose largest channel difference exceeds 32).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `underdetermined` | `underdetermined.py` | B: "The data does not determine the function" | 0.000 | 0.000% | match |
| `loss_regularization` | `loss_regularization.py` | B: "A penalty reshapes the whole surface" | 0.000 | 0.000% | match |
| `weight_decay_geometry` | `weight_decay_geometry.py` | B: "The same statement, drawn" | 0.000 | 0.000% | match |
| `shrinkage_factors` | `shrinkage_factors.py` | B: "The flat directions go first" | 0.000 | 0.000% | match |
| `penalty_sweep` | `penalty_sweep.py` | B: "What the coefficient costs" | 0.000 | 0.000% | match |
| `qnorm_contours` | `qnorm_contours.py` | B: "Other shapes of penalty" | 0.000 | 0.000% | match |
| `lasso_vs_ridge` | `lasso_vs_ridge.py` | B: "The two geometries side by side" | 0.000 | 0.000% | match |
| `implicit_gd` | `implicit_gd.py` | B: "Testing the claim" | 0.000 | 0.000% | match |
| `implicit_sgd` | `implicit_sgd.py` | B: "The two implicit penalties, measured" | 0.000 | 0.000% | match |

9 of 9 match; every figure is pixel-identical at 60 dpi (mean difference 0.000). Slowest script: implicit_gd (13 s); all 9 together about 27 s. Reruns are byte-identical (`CreationDate` is suppressed).

## Style fallback

`sci_style` is not installed here; `common.py` applies the plain fallback from CONVENTIONS.md (`axes.grid: False`, white figure and axes faces). This deck's figures were built with exactly that fallback: with it every figure matches pixel for pixel, whereas adding the SciencePlots `science` preset makes every figure about 6 px smaller with inward ticks (mean differences 12-30). If `sci_style` is ever installed, the output will differ from the slides for that reason.

## Helper modules

- `common.py` - style, palette, fonts, `save()` / `--out`
- `models.py` - truth, design matrices, ridge, rotated Hessians, the implicit-regularisation losses and descent helpers, copied unchanged from `src/common/models.py`
- Split: `fig02_weight_decay_geometry.py` -> `weight_decay_geometry.py` + `shrinkage_factors.py`; `fig03_lasso_vs_ridge.py` -> `qnorm_contours.py` + `lasso_vs_ridge.py`. Neither original draws random numbers, so the split cannot change any data.

## Figures with no script

| figure | deck | slide(s) | why |
|---|---|---|---|
| `PrinceReg9_1` | A | "The data does not determine the function" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_3` | A | "The same statement, drawn" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PrinceReg9_2` | A | "What the coefficient costs" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_5a` | A | "Other shapes of penalty" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_5b` | A | "Other shapes of penalty" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_5c` | A | "Other shapes of penalty" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_5d` | A | "Other shapes of penalty" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_6a` | A | "The two geometries side by side" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_6b` | A | "The two geometries side by side" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PrinceReg9_3` | A | "Testing the claim" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PrinceReg9_4` | A | "The two implicit penalties, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `PrinceReg9_5` | A | "The two predictions, measured" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_2` | A | "Equivariance, drawn" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_13` | A | "Residual connections" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_15` | A | "The same network, expanded" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop9_2` | B | "Equivariance, on an image" | book crop (`A_Textbook_Figures/figs_official/`) |

The B-variant copies of book figures in `B_Recreated_Figures/figs/` are byte-identical to the files in `A_Textbook_Figures/figs_official/`; no code in the repository writes them. 

TikZ drawn inside the `.tex` is not loaded with `\includegraphics` and needs no script.

No source found: none. Every code-like file in `figs/` that a deck loads is written by a module in `src/`.

## Moved to `figure_code/unused/main_class18/`

None: every module in `src/figures/` writes at least one figure a deck loads.
