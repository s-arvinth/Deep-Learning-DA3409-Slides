# Class 26 — BPTT and Bidirectional RNNs: figure code

Decks: `Class26_BPTT_Bidirectional_RNNs/A_Textbook_Figures/main.tex` and
`.../B_Recreated_Figures/main.tex`. The A deck loads two code-generated figures
(`running_example`, `space_vs_time`) from `A_Textbook_Figures/figs/`; those
files are byte-identical to the B `figs/` copies written by the B `src/`, so the
same scripts cover both decks.

Source: `B_Recreated_Figures/src/` (`common/style.py`, `draw.py`, `tape.py`,
`models.py`, `experiment.py`, `figures/fig00`-`fig08`).

Helpers here:

- `common.py` — style (fonts, sci_style fallback, palette, `save`) + `draw.py`
- `tape.py` — the reverse-mode autodiff tape, unchanged
- `models.py` — SimpleRNN, BPTT by hand, finite differences, recall task,
  timing, Bidirectional, influence maps; unchanged except `import tape`
- `experiment.py` — the cached experiment stages with the original seeds and
  parameters. The original cached all five stages together in
  `figs/_cache.pkl` and every measured figure ran all of them; here each
  stage is cached in `output/_cache_<stage>.pkl` and each script runs only
  the stage it needs (`check` -> bptt_check, `contrib` -> time_contributions,
  `timing` -> cost, `influence` -> influence_maps). Every stage seeds its own
  generators, so the results do not depend on which stages ran before; the
  freshly computed `check`, `contrib` and `influence` results are identical
  (max abs difference 0.0) to the ones in the original `figs/_cache.pkl`.

Split modules: `fig00_elman.py` -> `elman_network.py`, `elman_feedforward.py`;
`fig08_bidirectional.py` -> `bidirectional_goodfellow.py`,
`bidirectional_jurafsky.py`, `influence_maps.py`.

Verification: every script was run in a fresh `python3` process from a clean
`output/`, and its PDF compared with the deck's existing figure by rendering
both with `pdftoppm -r 60 -png` and taking the mean absolute pixel difference
(0-255) and the share of pixels differing by more than 32. Environment:
python3, numpy, matplotlib (sci_style not installed, so the fallback
rcParams from CONVENTIONS.md were used, as for the original build); fonts from
the repository's `fonts/`. Reference: `B_Recreated_Figures/figs/<name>.pdf`
(`running_example` and `space_vs_time` were also compared with
`A_Textbook_Figures/figs/`: identical result).

Caches were deleted before the run; the slowest script (time_contributions,
1500 training steps) takes about 3 s, so no stage needed a pre-built cache.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| bidirectional_goodfellow | bidirectional_goodfellow.py | B: "Both directions, drawn" | 0.000 | 0.000% | match |
| bidirectional_jurafsky | bidirectional_jurafsky.py | B: "Both directions, as two networks" | 0.000 | 0.000% | match |
| bptt_check | bptt_check.py | B: "The theorem, checked: the result" | 0.000 | 0.000% | match |
| bptt_flow | bptt_flow.py | B: "The two passes, drawn" | 0.000 | 0.000% | match |
| cost | cost.py | B: "The cost, measured" | 2.599 | 1.671% | MISMATCH (explained: wall-clock timing, see below) |
| design_patterns | design_patterns.py | B: "Three design patterns" | 0.000 | 0.000% | match |
| elman_feedforward | elman_feedforward.py | B: "The same network, as one feedforward step" | 0.000 | 0.000% | match |
| elman_network | elman_network.py | B: "The simple recurrent network, drawn" | 0.000 | 0.000% | match |
| influence_maps | influence_maps.py | B: "Both directions, measured" | 0.000 | 0.000% | match |
| running_example | running_example.py | A, B: "How the recurrent network reads the sentence" | 0.000 | 0.000% | match |
| sequence_tasks | sequence_tasks.py | B: "What natural language processing asks of a model" | 0.000 | 0.000% | match |
| space_vs_time | space_vs_time.py | A, B: "The idea: from sharing across space to sharing across time" | 0.000 | 0.000% | match |
| time_contributions | time_contributions.py | B: "The terms of the sum: the result" | 0.000 | 0.000% | match |
| unfolding | unfolding.py | B: "Unfolding, drawn" | 0.000 | 0.000% | match |
| window_fails | window_fails.py | B: "Why the tools we have fall short" | 0.000 | 0.000% | match |

All 15 scripts: 14 match, 0 near, 1 MISMATCH (`cost`, explained).

### `cost`: why it cannot match, and the check that the code is right

`cost.pdf` plots wall-clock seconds of forward and backward passes
(`models.timing`, `time.perf_counter`). Timings depend on the machine and its
load and cannot be fixed by a seed, so a fresh run draws the same axes, labels,
stored-state panel and slope-1 guide but slightly different timing curves
(fresh run here vs. original cache, forward ms at tau = 320: 10.7 vs 14.2).
To confirm the plotting code is an exact port, the script was re-run in a
scratch copy with the original `timing` stage from
`B_Recreated_Figures/figs/_cache.pkl` placed in `output/_cache_timing.pkl`:
the output is then pixel-identical (mean diff 0.000, >32 share 0.000%).
To reproduce the slide figure exactly, do the same (load
`pickle.load(open(".../figs/_cache.pkl","rb"))["timing"]` and pickle it to
`output/_cache_timing.pkl`) before running `python cost.py`.

## Figures with no script

| figure | deck | why |
|---|---|---|
| G10_3, G10_4, G10_5, G10_11 | A | book crops (`figs_official/`, cut by `A_Textbook_Figures/crop_figures.py`, no figure script needed) |
| J14_1, J14_2, J14_3, J14_4, J14_5, J14_11, J14_15 | A | book crops (`figs_official/`, same) |

No source found: none. TikZ: none loaded via `\includegraphics`.

## Unused original modules

Copied (not moved) to `figure_code/unused/main_class26/`:

- `fig07_truncated_bptt.py` — writes `truncated_bptt.pdf`, which neither deck
  loads (the `trunc` stage is still in `experiment.py`, unchanged, but no
  script here runs it).
