# Class 28 — figure scripts

Code-generated figures of `Class28_LSTM_Gated_RNNs` (A_Textbook_Figures and B_Recreated_Figures), one script each.
Each `python <name>.py` writes `output/<name>.pdf` (`--out DIR` to write elsewhere), with the code, data,
parameters and seeds of the original `B_Recreated_Figures/src/`.

## Verification

Every script was run in a fresh process from an empty `output/` (caches deleted first); each output was
rendered with `pdftoppm -r 60 -png` next to the deck's existing figure (`figs/<name>.pdf`) and compared
by mean absolute pixel difference (0–255) and the share of pixels differing by more than 32.

Result: 11 match, 0 near, 0 MISMATCH.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `gradient_paths` | `gradient_paths.py` | The three paths, drawn (A/B) | 0.000 | 0.00% | match |
| `power_iteration` | `power_iteration.py` | The power method: the result (B) | 0.000 | 0.00% | match |
| `composition` | `composition.py` | The composition, drawn (B) | 0.000 | 0.00% | match |
| `gradient_lag` | `gradient_lag.py` | The gradient by lag, measured (B) | 0.000 | 0.00% | match |
| `leaky_unit` | `leaky_unit.py` | The leaky unit is a running average (B) | 0.000 | 0.00% | match |
| `lstm_cell` | `lstm_cell.py` | The LSTM cell, drawn (B) | 0.000 | 0.00% | match |
| `lstm_block` | `lstm_block.py` | The LSTM cell, as a block diagram (B) | 0.000 | 0.00% | match |
| `dropin_modules` | `dropin_modules.py` | Gated units are drop-in modules (B) | 0.000 | 0.00% | match |
| `long_dependency` | `long_dependency.py` | Three cells on one task (B) | 0.000 | 0.00% | match |
| `clipping` | `clipping.py` | Clipping at a cliff, drawn (B) | 0.000 | 0.00% | match |
| `explicit_memory` | `explicit_memory.py` | Where the gate leads: explicit memory, drawn (B) | 0.000 | 0.00% | match |

Experiment time: The three cells train in about 35 s together (first script that needs them: gradient_lag or long_dependency).

The training loops write a checkpoint (`output/_ckpt_*.pkl`: parameters, Adam moments, generator state,
loss history) and resume from it if interrupted; a resumed run was checked to give a bit-identical cache
to an uninterrupted one.

## Figures with no script

- `G10_15` — The composition, drawn (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `G10_16` — The LSTM cell, as a block diagram (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `G10_17` — Clipping at a cliff, drawn (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `G10_18` — Where the gate leads: explicit memory, drawn (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_13` — The LSTM cell, drawn (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_14` — Gated units are drop-in modules (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.

No source found: none (every code-generated-looking figure loaded by the deck is written by `src/`).
TikZ drawn inside the .tex needs no script and is not listed.

## Helper modules

- `common.py` — style: Fira fonts from ../../fonts, sci_style fallback, palette, rcParams, tick thinning, `save()` -> output/<name>.pdf or `--out DIR` (from src/common/style.py)
- `tape.py` — the reverse-mode autodiff tape and Adam (src/common/tape.py), plus `snapshot()`/`restore()` for resumable training
- `models.py` — RNN/LSTM/GRU cells, the recall task, training (now with an optional checkpoint), `grad_by_lag`, and the closed-form pictures (power iteration, linear-tanh composition, leaky unit, cliff loss) (src/common/models.py)
- `experiment.py` — the shared experiment, same seeds; one cache stage per cell: output/_cache_RNN.pkl, _cache_LSTM.pkl, _cache_GRU.pkl (checkpoint output/_ckpt_<cell>.pkl while training)
- `draw.py` — colours and the arrow helper used by gradient_paths (src/common/draw.py)

## Split multi-figure modules

- none (every original module wrote one figure).

## Unused original code (copied to `figure_code/unused/main_class28/`)

- `src/figures/fig05_gate_activity.py` — writes `gate_activity.pdf`, which neither variant's main.tex loads

## Other notes

- `gradient_paths` is loaded by both A_Textbook_Figures (from its own figs/) and B_Recreated_Figures; the two PDFs are byte-identical and the script's output matches both.
- `src/build_all.py` is a driver that only calls the figure modules; not ported or copied.
