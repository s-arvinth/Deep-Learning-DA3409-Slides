# Class 6 — Shallow Neural Networks: Classification: figure code

Source: `Class06_Shallow_NN_Classification/B_Recreated_Figures/src/figures/figNN_<name>.py` (one module per figure), each ported to `<NAME>.py`. `src/common/{style,data,models}.py` are merged into the single helper `common.py` (unchanged code; `save()` now writes to `output/` or `--out`).

Seeds, as in the original: `discriminant` `two_gaussians(seed=3)`; `lsq_vs_logistic` `default_rng(5)` for the extra points, drawn after `two_gaussians(seed=3)`; `binary_model` `default_rng(6)`; `softmax_posteriors` `three_gaussians(seed=8)`; `gaussian_ml` `regression_cloud(seed=4)`; `cross_entropy_kl` `default_rng(2)`; `feature_space` `noisy_xor(seed=11)` and `_train` tries `default_rng(0..59)` in order, keeping the first seed that gives a non-degenerate fit (same loop as the original).

## Verification

Every script was run in a fresh `python3 <name>.py` process from an empty
`output/` (matplotlib 3.10.9, numpy 2.2.6, no `sci_style` installed, so the
fallback in `common.py` was in effect). Each output and the deck's
`B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png`
and compared: mean absolute pixel difference (0-255) and share of pixels
differing by more than 32. Reruns are byte-identical (`CreationDate` stripped).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `discriminant.pdf` | `discriminant.py` | "Discriminant functions" | 0.000 | 0.000% | match |
| `lsq_vs_logistic.pdf` | `lsq_vs_logistic.py` | "Why not just do least squares?" | 0.000 | 0.000% | match |
| `gaussian_posterior.pdf` | `gaussian_posterior.py` | "The sigmoid, seen in the data" | 0.000 | 0.000% | match |
| `gaussian_ml.pdf` | `gaussian_ml.py` | "You already know one instance" | 0.000 | 0.000% | match |
| `bernoulli.pdf` | `bernoulli.py` | "Step 1: the Bernoulli distribution" | 0.000 | 0.000% | match |
| `sigmoid.pdf` | `sigmoid.py` | "Step 2: the logistic sigmoid" | 0.000 | 0.000% | match |
| `binary_model.pdf` | `binary_model.py` | "The binary model, end to end" | 0.000 | 0.000% | match |
| `saturation.pdf` | `saturation.py` | "Why not squared error on probabilities?" | 0.000 | 0.000% | match |
| `categorical.pdf` | `categorical.py` | "Step 1: the categorical distribution" | 0.000 | 0.000% | match |
| `softmax_posteriors.pdf` | `softmax_posteriors.py` | "Softmax posteriors over the input space" | 0.000 | 0.000% | match |
| `multiclass_model.pdf` | `multiclass_model.py` | "The multi-class model on a scalar input" | 0.000 | 0.000% | match |
| `output_distributions.pdf` | `output_distributions.py` | "One principle, any output type" | 0.000 | 0.000% | match |
| `cross_entropy_kl.pdf` | `cross_entropy_kl.py` | "The same statement, drawn" | 0.000 | 0.000% | match |
| `feature_space.pdf` | `feature_space.py` | "Fixed features vs. learned features, again" | 0.000 | 0.000% | match |

All 14 scripts: match (14), near (0), MISMATCH (0).

## Style note: the `sci_style` fallback

The figures in `figs/` were rendered with the SciencePlots `science` preset
active (`import sci_style`), which is not in the repository or installed here.
The plain fallback in CONVENTIONS.md (`axes.grid`/`facecolor` only) did not
reproduce them: page sizes were off by ~10 pt, ticks pointed outwards, minor
ticks were missing (mean diff 9-24, share 7-14% on every figure
tried; MISMATCH). `common.py`
therefore restates the preset values the figures were evidently built with
(`SCIENCE_RC`: inward ticks on all four sides, minor ticks, 0.5 pt tick/axis
widths, `savefig.pad_inches` 0.05, frameless legends, and
`axes.formatter.use_mathtext=True`) before the convention's fallback block and
the deck's own rcParams. With that, every figure matches pixel for pixel. If a
real `sci_style` is importable it is used instead and `SCIENCE_RC` is skipped.

## Font note

`B_Recreated_Figures/fonts/` in this deck has no `FiraSans-Bold.ttf`, so when
`figs/` was rendered every `$\mathbf{...}$` (e.g. the bold **w**, **a**, **x**,
**Sigma** in `discriminant`, `categorical`, `gaussian_ml`, `softmax_posteriors`,
`feature_space`) fell back to Fira Sans Regular. Registering the whole root
`fonts/` folder made those glyphs genuinely bold (mean diff up to 0.38, share
0.52% on `discriminant`). `common.py` therefore registers, from the root
`fonts/`, only the six faces the deck folder had (`FACES`), which reproduces
the slide figures exactly. Add `FiraSans-Bold.ttf` to `FACES` if bold math is
wanted in a future re-render (that would change the figures).

## Figures with no script

- A deck (`A_Textbook_Figures/main.tex`): Bishop5_1, Bishop5_4a-b, Bishop5_13a-b, Bishop5_14a-b, Bishop5_15a-b, LossBern, LossBinaryClassification, LossCategorical, LossCrossEntropy, LossDataTypes, LossLogisticSigmoid, LossMultiClassClassification, LossNormalRegression — book crops in `figs_official/`. No script.
- No source found: none. (`B_Recreated_Figures/figs_official/` also holds copies of the book crops; the B deck loads none of them.)

## Unused code

No figure module is unused: all 14 `figures/fig*.py` write a figure the B deck loads. The driver `src/build_all.py` writes no figure itself and was not copied to `figure_code/unused/main_class6/`.
