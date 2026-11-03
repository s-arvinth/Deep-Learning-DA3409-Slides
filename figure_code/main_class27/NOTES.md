# Class 27 — figure scripts

Code-generated figures of `Class27_Encoder_Decoder_Seq2Seq` (A_Textbook_Figures and B_Recreated_Figures), one script each.
Each `python <name>.py` writes `output/<name>.pdf` (`--out DIR` to write elsewhere), with the code, data,
parameters and seeds of the original `B_Recreated_Figures/src/`.

## Verification

Every script was run in a fresh process from an empty `output/` (caches deleted first); each output was
rendered with `pdftoppm -r 60 -png` next to the deck's existing figure (`figs/<name>.pdf`) and compared
by mean absolute pixel difference (0–255) and the share of pixels differing by more than 32.

Result: 14 match, 0 near, 0 MISMATCH.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `generation` | `generation.py` | Generating, one token at a time (B) | 0.000 | 0.00% | match |
| `condition_input` | `condition_input.py` | Two ways to hand over a vector: as an extra input (B) | 0.000 | 0.00% | match |
| `condition_encdec` | `condition_encdec.py` | Two ways to hand over a vector: as the first state (B) | 0.000 | 0.00% | match |
| `seq2seq_full` | `seq2seq_full.py` | The whole model on the example (B) | 0.000 | 0.00% | match |
| `seq2seq_unrolled` | `seq2seq_unrolled.py` | The encoder--decoder, unrolled (B) | 0.000 | 0.00% | match |
| `teacher_forcing_book` | `teacher_forcing_book.py` | Teacher forcing, on the example (B) | 0.000 | 0.00% | match |
| `teacher_forcing` | `teacher_forcing.py` | Training against inference, on the trained model (B) | 0.000 | 0.00% | match |
| `bottleneck_book` | `bottleneck_book.py` | Everything through one vector (B) | 0.000 | 0.00% | match |
| `bottleneck` | `bottleneck.py` | The bottleneck, measured: the result (B) | 0.000 | 0.00% | match |
| `attention_top` | `attention_top.py` | Attention, drawn: a context per step (B) | 0.000 | 0.00% | match |
| `attention_full` | `attention_full.py` | Attention, drawn: one decoder step in full (B) | 0.000 | 0.00% | match |
| `context_hull` | `context_hull.py` | Where the context can be (B) | 0.000 | 0.00% | match |
| `beam_toy` | `beam_toy.py` | Greedy against beam, on a toy vocabulary (B) | 0.000 | 0.00% | match |
| `beam_search` | `beam_search.py` | Beam search, run on the trained model (B) | 0.000 | 0.00% | match |

Experiment time: Both models train in about 30 s together (first script that needs them: teacher_forcing, bottleneck, context_hull or beam_search).

The training loops write a checkpoint (`output/_ckpt_*.pkl`: parameters, Adam moments, generator state,
loss history) and resume from it if interrupted; a resumed run was checked to give a bit-identical cache
to an uninterrupted one.

## Figures with no script

- `G10_12` — Two ways to hand over a vector: as the first state (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `G10_9` — Two ways to hand over a vector: as an extra input (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J13_7` — Greedy against beam, on a toy vocabulary (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_17` — The whole model on the example (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_18` — The encoder--decoder, unrolled (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_19` — Teacher forcing, on the example (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_20` — Everything through one vector (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_21` — Attention, drawn: a context per step (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_22` — Attention, drawn: one decoder step in full (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.
- `J14_9` — Generating, one token at a time (A): book crop from `A_Textbook_Figures/figs_official/`, used as given.

No source found: none (every code-generated-looking figure loaded by the deck is written by `src/`).
TikZ drawn inside the .tex needs no script and is not listed.

## Helper modules

- `common.py` — style: Fira fonts from ../../fonts, sci_style fallback, palette, rcParams, tick thinning, `save()` -> output/<name>.pdf or `--out DIR` (from src/common/style.py)
- `tape.py` — the reverse-mode autodiff tape and Adam (src/common/tape.py), plus `snapshot()`/`restore()` for resumable training
- `models.py` — `Seq2Seq`, the reversal task, training (now with an optional checkpoint), greedy/beam decoding (src/common/models.py)
- `experiment.py` — the shared experiment, same seeds; one cache stage per model: output/_cache_fixed.pkl, output/_cache_attention.pkl (checkpoint output/_ckpt_<stage>.pkl while training)
- `draw.py` — unrolled-network helpers used by teacher_forcing and condition_* (src/common/draw.py)
- `bookdraw.py` — the book-layout helpers (src/common/bookdraw.py)

## Split multi-figure modules

- `src/figures/fig06_beam_search.py` -> beam_toy, beam_search
- `src/figures/fig08_conditioning.py` -> condition_input, condition_encdec
- `src/figures/fig09_seq2seq_book.py` -> seq2seq_full, seq2seq_unrolled
- `src/figures/fig12_attention_book.py` -> attention_top, attention_full

## Unused original code (copied to `figure_code/unused/main_class27/`)

- none: every figure module in `src/figures/` writes a figure the deck loads.

## Other notes

- `src/build_all.py` is a driver that only calls the figure modules; it writes no figure of its own and was not ported or copied.
