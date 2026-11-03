# Class 14 - Convergence: figure code

Scripts for every code-generated figure loaded by `Class14_Convergence/A_Textbook_Figures/main.tex` or `Class14_Convergence/B_Recreated_Figures/main.tex`. Each `<name>.py` is a port of the module in `Class14_Convergence/B_Recreated_Figures/src/figures/` that wrote `figs/<name>.pdf` (path in each header), with the same data, parameters, seeds and style.

## Verification

Every script was run in a fresh process (`python3 <name>.py`) from an empty `output/`; no caches are used. Each `output/<name>.pdf` and the deck's `B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png` and compared pixel by pixel (mean absolute difference on the 0-255 scale; share of pixels whose largest channel difference exceeds 32).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `convergence_rates` | `convergence_rates.py` | B: "Two rates, drawn" | 0.000 | 0.000% | match |
| `eigen_decay` | `eigen_decay.py` | B: "One learning rate, many curvatures" | 0.000 | 0.000% | match |
| `well_conditioned` | `well_conditioned.py` | B: "When every direction curves alike" | 0.000 | 0.000% | match |
| `ellipse_geometry` | `ellipse_geometry.py` | B: "Why the shape matters for descent" | 0.000 | 0.000% | match |
| `valley` | `valley.py` | B: "The valley" | 0.000 | 0.000% | match |
| `momentum_regimes` | `momentum_regimes.py` | B: "The two regimes" | 0.000 | 0.000% | match |
| `momentum_multiplier` | `momentum_multiplier.py` | B: "The same two regimes, measured" | 0.000 | 0.000% | match |
| `nesterov_path` | `nesterov_path.py` | B: "Nesterov on the same valley" | 0.000 | 0.000% | match |
| `momentum_error` | `momentum_error.py` | B: "What that costs in iterations" | 0.000 | 0.000% | match |
| `noise_floor` | `noise_floor.py` | B: "The noise floor" | 0.000 | 0.000% | match |
| `lr_schedules` | `lr_schedules.py` | B: "The schedules in use" | 0.000 | 0.000% | match |
| `bias_correction` | `bias_correction.py` | B: "The correction, drawn" | 0.000 | 0.000% | match |
| `optimiser_compare` | `optimiser_compare.py` | B: "Five update rules, by family" | 0.000 | 0.000% | match |
| `per_parameter` | `per_parameter.py` | B: "The problem, and three responses" | 0.000 | 0.000% | match |

14 of 14 match; every figure is pixel-identical at 60 dpi (mean difference 0.000). Slowest script: noise_floor (2 s); all 14 together about 21 s. Reruns are byte-identical (`CreationDate` is suppressed).

## Style fallback

`sci_style` is not installed here. The plain fallback in CONVENTIONS.md (`axes.grid`, `figure.facecolor`, `axes.facecolor`) does NOT reproduce this deck's figures: with it every figure came out about 6 px larger at 60 dpi, with outward ticks and plain-text tick labels (mean differences 12-21, MISMATCH throughout). The PDFs in `figs/` were rendered with the SciencePlots `science` preset active: inward ticks on all four sides, minor ticks, 0.05 in save padding, 1.0 default line width, and tick labels typeset as mathtext (`axes.formatter.use_mathtext`; visible in the PDFs as glyph-by-glyph text runs). `common.py` therefore applies the conventions' block plus those rcParams when `import sci_style` fails. With that fallback every figure matches exactly; the font/usetex keys of the preset are omitted because `use()` overrides them.

## Helper modules

- `common.py` - style, palette, fonts, `save()` / `--out`
- `models.py` - the quadratic bowl, the optimisers, the schedules and the optimal constants, copied unchanged from `src/common/models.py`
- `momentum_error.py` imports `LAM` and `runs()` from `nesterov_path.py`, exactly as `fig15` imported them from `fig07` (no other figure script imports a figure script).

## Outputs of the original modules that no deck loads (not reproduced)

- `momentum_paths.pdf` (written by `fig07_momentum_paths.py` next to `nesterov_path.pdf`; no deck loads it, so `nesterov_path.py` keeps only the `("mom", "nes")` pair)
- `optimiser_cost.pdf` (written by `fig11_optimiser_compare.py` next to `optimiser_compare.pdf`; no deck loads it, so `optimiser_compare.py` drops `_cost()`)

## Figures with no script

| figure | deck | slide(s) | why |
|---|---|---|---|
| `Bishop7_3` | A | "The valley" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop7_4` | A | "The two regimes" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop7_5` | A | "The two regimes" | book crop (`A_Textbook_Figures/figs_official/`) |
| `Bishop7_6` | A | "Momentum on the valley" | book crop (`A_Textbook_Figures/figs_official/`) |
| `TrainNesterov` | A | "Nesterov momentum" | book crop (`A_Textbook_Figures/figs_official/`) |
| `TrainADAM` | A | "The problem, and three responses" | book crop (`A_Textbook_Figures/figs_official/`) |
| `TrainMomentum` | A | "Momentum on a noisy problem" | book crop (`A_Textbook_Figures/figs_official/`) |

TikZ drawn inside the `.tex` is not loaded with `\includegraphics` and needs no script.

No source found: none. Every code-like file in `figs/` that a deck loads is written by a module in `src/`.

## Moved to `figure_code/unused/main_class14/`

Copied (not moved; `src/` is untouched) because they write no figure that any deck loads:

- `fig04_condition_number.py` (writes `condition_number.pdf`, which no deck loads)
