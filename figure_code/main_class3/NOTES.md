# Class 3 — Perceptrons and the PLA: figure code

Source: `Class03_Perceptrons_PLA/B_Recreated_Figures/src/make_figures.py`, split into one `<name>.py` per figure (`fig_update` -> `update_step.py`, the rest by figure name). Shared rcParams, palette, `save()`, `scatter_PN`, `boundary`, `pla`, the AND error surface (`AND_X`, `AND_Y`, `AND_TH`, `and_error`) and `pla_fixed_threshold` are in `common.py`.

Seeds, as in the original: `pla_run` draws the data with `np.random.default_rng(4)` (P then N) and picks misclassified points with `np.random.default_rng(3)`; `error_surface` and `update_ws` are deterministic (the fixed-threshold PLA always takes the first wrong pattern).

## Verification

Every script was run in a fresh `python3 <name>.py` process from an empty
`output/` (matplotlib 3.10.9, numpy 2.2.6, no `sci_style` installed, so the
fallback in `common.py` was in effect). Each output and the deck's
`B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png`
and compared: mean absolute pixel difference (0-255) and share of pixels
differing by more than 32. Reruns are byte-identical (`CreationDate` stripped).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `regions.pdf` | `regions.py` | "Weights tilt the separating line" | 0.000 | 0.000% | match |
| `weights_bias.pdf` | `weights_bias.py` | "What the weights and the bias each control" | 0.000 | 0.000% | match |
| `update_step.pdf` | `update_step.py` | "Why the update works" | 0.000 | 0.000% | match |
| `pla_run.pdf` | `pla_run.py` | "The algorithm running" | 0.000 | 0.000% | match |
| `error_surface.pdf` | `error_surface.py` | "The error surface" | 0.000 | 0.000% | match |
| `update_ws.pdf` | `update_ws.py` | "The update, seen in weight space" | 0.000 | 0.000% | match |
| `duality.pdf` | `duality.py` | "Two dual viewpoints" | 0.000 | 0.000% | match |
| `xor.pdf` | `xor.py` | "The limit: XOR and non-separable data" | 0.000 | 0.000% | match |

All 8 scripts: match (8), near (0), MISMATCH (0).

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

- A deck (`A_Textbook_Figures/main.tex`): bias, classic_perceptron, duality, error_surface, pla_input, sep_perceptron, weight_update — book crops in `A_Textbook_Figures/figs_official/` (PNG). No script. (`duality` and `error_surface` share a name with B-deck figures, but the A deck loads the PNG book crops.)
- B deck `classic_perceptron` (`B_Recreated_Figures/figs/classic_perceptron.png`): an image taken from elsewhere, byte-identical to the A deck's `figs_official/classic_perceptron.png`. Not written by any code. No script.
- No source found: none.

## Unused code

No module was copied to `figure_code/unused/main_class3/`: the only module, `make_figures.py`, writes used figures.
Its function `fig_margin()` writes `figs/margin.pdf`, which neither deck loads; it was not ported (the original stays in `src/make_figures.py`).
