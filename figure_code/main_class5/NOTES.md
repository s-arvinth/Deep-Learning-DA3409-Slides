# Class 5 — Shallow Neural Networks: Regression: figure code

Source: `Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/figNN_<name>.py` (one module per figure, `NAME` = output name), each ported to `<NAME>.py`. `src/common/{style,data,models}.py` are merged into the single helper `common.py` (unchanged code; `save()` now writes to `output/` or `--out`).

Seeds, as in the original: `linear_fit`/`loss_surface` use `linear_data(seed=5)`, `conditional_mean` uses `gaussian_cloud(seed=2)`, `family` `default_rng(11)`, `regions` `default_rng(M)` for M = 2, 5, 12, `fits` `default_rng(7)`; the rest are deterministic.

## Verification

Every script was run in a fresh `python3 <name>.py` process from an empty
`output/` (matplotlib 3.10.9, numpy 2.2.6, no `sci_style` installed, so the
fallback in `common.py` was in effect). Each output and the deck's
`B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png`
and compared: mean absolute pixel difference (0-255) and share of pixels
differing by more than 32. Reruns are byte-identical (`CreationDate` stripped).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `linear_fit.pdf` | `linear_fit.py` | "Linear regression, two views" | 0.000 | 0.000% | match |
| `loss_surface.pdf` | `loss_surface.py` | "Training: minimise the error" | 0.000 | 0.000% | match |
| `basis_functions.pdf` | `basis_functions.py` | "Beyond lines: fixed basis functions" | 0.000 | 0.000% | match |
| `conditional_mean.pdf` | `conditional_mean.py` | "Why least squares? The probabilistic view" | 0.000 | 0.000% | match |
| `relu.pdf` | `relu.py` | "The activation function: ReLU" | 0.000 | 0.000% | match |
| `buildup.pdf` | `buildup.py` | "Computing the function, step by step" | 0.000 | 0.000% | match |
| `family.pdf` | `family.py` | "The function family: piecewise linear" | 0.000 | 0.000% | match |
| `learned_basis.pdf` | `learned_basis.py` | "Hidden units are basis functions that adapt" | 0.000 | 0.000% | match |
| `two_outputs.pdf` | `two_outputs.py` | "More than one output" | 0.000 | 0.000% | match |
| `two_inputs.pdf` | `two_inputs.py` | "Each unit is a hinge over the input plane" | 0.000 | 0.000% | match |
| `buildup2d.pdf` | `buildup2d.py` | "Two inputs, computed step by step" | 0.000 | 0.000% | match |
| `regions.pdf` | `regions.py` | "How many regions?" | 0.000 | 0.000% | match |
| `activations.pdf` | `activations.py` | "Other activation functions" | 0.000 | 0.000% | match |
| `ua_construction.pdf` | `ua_construction.py` | "The construction, drawn" | 0.000 | 0.000% | match |
| `approximate.pdf` | `approximate.py` | "Using the theorem: how fast does the error fall?" | 0.000 | 0.000% | match |
| `fits.pdf` | `fits.py` | "It works: shallow networks fit real data" | 0.000 | 0.000% | match |

All 16 scripts: match (16), near (0), MISMATCH (0).

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

## Figures with no script

- A deck (`A_Textbook_Figures/main.tex`): Bishop4_1, Bishop4_2a-c, Bishop4_3, Bishop4_5, Bishop6_9, Bishop6_10a-d, Bishop6_12a-f, Bishop6_17a-b, ShallowActivations, ShallowApproximate, ShallowBuildUp, ShallowBuildUp2D, ShallowFunctions, ShallowHyperplanes, ShallowNet, ShallowNetThreeInputsTwoOutputs, ShallowNetTwoInputs, ShallowNetTwoOutputs, ShallowReLU, ShallowRegions, ShallowTerminology, SupervisedLinear, SupervisedOpt — book crops (Bishop & Bishop 2024, Prince 2023) in `figs_official/`. No script.
- B deck `Bishop4_3` (`B_Recreated_Figures/figs/Bishop4_3.pdf`): book crop, byte-identical to `A_Textbook_Figures/figs_official/Bishop4_3.pdf` (the src README says so too). No script.
- No source found: none.

## Unused code

No figure module is unused: all 16 `figures/fig*.py` write a figure the B deck loads. The drivers `src/build_all.py` and `src/make_figures.py` (a deprecated wrapper around `build_all`) write no figure themselves; they are superseded by running each `<name>.py` and were not copied to `figure_code/unused/main_class5/`.
