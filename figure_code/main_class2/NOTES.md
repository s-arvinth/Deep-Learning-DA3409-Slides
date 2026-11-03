# Class 2 — Introduction to Neural Networks: figure code

Source: `Class02_Intro_Neural_Networks/B_Recreated_Figures/src/make_figures.py` (one script, seven figures), split into one `<name>.py` per figure. Shared rcParams, palette, `save()` and the cube helpers used by `cube_or`/`cube_majority` (`_cube`, `_plane`, `fig_cube`) are in `common.py`. No random numbers are drawn in any of these figures, so there are no seeds to fix.

## Verification

Every script was run in a fresh `python3 <name>.py` process from an empty
`output/` (matplotlib 3.10.9, numpy 2.2.6, no `sci_style` installed, so the
fallback in `common.py` was in effect). Each output and the deck's
`B_Recreated_Figures/figs/<name>.pdf` were rendered with `pdftoppm -r 60 -png`
and compared: mean absolute pixel difference (0-255) and share of pixels
differing by more than 32. Reruns are byte-identical (`CreationDate` stripped).

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `action_potential.pdf` | `action_potential.py` | "Firing: an all-or-nothing threshold" | 0.000 | 0.000% | match |
| `approximation.pdf` | `approximation.py` | "Simple units add up to complicated functions" | 0.000 | 0.000% | match |
| `step_activation.pdf` | `step_activation.py` | "The activation is a step function" | 0.000 | 0.000% | match |
| `logic_2d.pdf` | `logic_2d.py` | "The same gates, drawn in input space" | 0.000 | 0.000% | match |
| `cube_or.pdf` | `cube_or.py` | "A unit splits input space into half-spaces" | 0.000 | 0.000% | match |
| `cube_majority.pdf` | `cube_majority.py` | "Parallel planes, different thresholds" | 0.000 | 0.000% | match |
| `separable_fraction.pdf` | `separable_fraction.py` | "How rare is linear separability?" | 0.000 | 0.000% | match |

All 7 scripts: match (7), near (0), MISMATCH (0).

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

- A deck (`A_Textbook_Figures/main.tex`): abstract_neuron, and_or, mp_unit, sep_or, sep_planes, step_fn, taylor_net — book crops in `A_Textbook_Figures/figs_official/` (PNG). No script.
- `figs_official/logic_real.png` is present but not loaded by either deck.
- No source found: none (every code-looking figure in `B_Recreated_Figures/figs/` is written by `make_figures.py`).

## Unused code

None. `make_figures.py` writes only figures that the B deck loads; nothing was copied to `figure_code/unused/main_class2/`.
