# Class 29 — figure scripts

Code-generated figures of `Class29_RNN_NLP_Tasks` (A_Textbook_Figures and B_Recreated_Figures), one script each.
Each `python <name>.py` writes `output/<name>.pdf` (`--out DIR` to write elsewhere), with the code, data,
parameters and seeds of the original `B_Recreated_Figures/src/`.

## Verification

Every script was run in a fresh process from an empty `output/` (caches deleted first); each output was
rendered with `pdftoppm -r 60 -png` next to the deck's existing figure (`figs/<name>.pdf`) and compared
by mean absolute pixel difference (0–255) and the share of pixels differing by more than 32.

Result: 19 match, 0 near, 0 MISMATCH.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| `rnn_lm_training` | `rnn_lm_training.py` | Training by self-supervision (B) | 0.000 | 0.00% | match |
| `char_lm_curve` | `char_lm_curve.py` | A language model, trained: the perplexity (B) | 0.000 | 0.00% | match |
| `char_lm_samples` | `char_lm_samples.py` | A language model, trained: the samples (B) | 0.000 | 0.00% | match |
| `generation` | `generation.py` | Generation, drawn (B) | 0.000 | 0.00% | match |
| `pos_tagging` | `pos_tagging.py` | Sequence labelling: a label at every step (B) | 0.000 | 0.00% | match |
| `seq_classification` | `seq_classification.py` | Sequence classification, drawn (B) | 0.000 | 0.00% | match |
| `stacked_bidirectional` | `stacked_bidirectional.py` | RNNs as modules (B) | 0.000 | 0.00% | match |
| `deep_a` | `deep_a.py` | Three ways to make an RNN deep (B) | 0.000 | 0.00% | match |
| `deep_b` | `deep_b.py` | Three ways to make an RNN deep (B) | 0.000 | 0.00% | match |
| `deep_c` | `deep_c.py` | Three ways to make an RNN deep (B) | 0.000 | 0.00% | match |
| `bidir_classification` | `bidir_classification.py` | Bidirectional classification, drawn (B) | 0.000 | 0.00% | match |
| `gm_complete` | `gm_complete.py` | The two graphs, drawn (B) | 0.000 | 0.00% | match |
| `gm_state` | `gm_state.py` | The two graphs, drawn (B) | 0.000 | 0.00% | match |
| `recursive_tree` | `recursive_tree.py` | The tree-shaped cousin: recursive networks (B) | 0.000 | 0.00% | match |
| `four_architectures` | `four_architectures.py` | Four architectures (B) | 0.000 | 0.00% | match |
| `arch_labelling` | `arch_labelling.py` | (a) Sequence labelling (B) | 0.000 | 0.00% | match |
| `arch_classification` | `arch_classification.py` | (b) Sequence classification (B) | 0.000 | 0.00% | match |
| `arch_lm` | `arch_lm.py` | (c) Language modelling (B) | 0.000 | 0.00% | match |
| `arch_encdec` | `arch_encdec.py` | (d) Encoder--decoder (B) | 0.000 | 0.00% | match |

Experiment time: The LM stage takes about 20 s and the tagger stage about 4 s here. The original README's "about four minutes" covered all five stages of src/common/experiment.py; the three no slide uses (`lm_untied`, `depth`, `cls`) were not ported. Each stage seeds its own generators, so dropping them leaves the LM and tagger results unchanged (confirmed: both figures match exactly).

The training loops write a checkpoint (`output/_ckpt_*.pkl`: parameters, Adam moments, generator state,
loss history) and resume from it if interrupted; a resumed run was checked to give a bit-identical cache
to an uninterrupted one.

## Figures with no script

- none: every figure this deck loads is code-generated.

No source found: none (every code-generated-looking figure loaded by the deck is written by `src/`).
TikZ drawn inside the .tex needs no script and is not listed.

## Helper modules

- `common.py` — style: Fira fonts from ../../fonts, sci_style fallback, palette, rcParams, tick thinning, `save()` -> output/<name>.pdf or `--out DIR` (from src/common/style.py)
- `tape.py` — the reverse-mode autodiff tape and Adam (src/common/tape.py), plus `snapshot()`/`restore()` for resumable training
- `models.py` — cells, `CharLM`, perplexity, sampling, the tagger and grammar (src/common/models.py; the text is now read from data/magi.txt; `train_lm` takes an optional checkpoint)
- `experiment.py` — the two stages a slide uses, same seeds: output/_cache_lm.pkl and output/_cache_tagger.pkl (checkpoint output/_ckpt_lm.pkl every 50 steps while training)
- `bookdraw.py` — the book-layout helpers (src/common/bookdraw.py)

## Split multi-figure modules

- `src/figures/fig02_char_lm.py` -> char_lm_curve, char_lm_samples
- `src/figures/fig08_four_architectures.py` -> four_architectures, arch_labelling, arch_classification, arch_lm, arch_encdec
- `src/figures/fig11_seq_classification.py` -> seq_classification, bidir_classification
- `src/figures/fig12_deep_ways.py` -> deep_a, deep_b, deep_c
- `src/figures/fig13_graphical.py` -> gm_complete, gm_state

## Unused original code (copied to `figure_code/unused/main_class29/`)

- `src/common/draw.py` — the older unrolled-network helpers; no Class 29 figure imports them

## Other notes

- `data/magi.txt` is copied unchanged from src/data/magi.txt.
- Only Class 29's B_Recreated_Figures variant exists in this copy.
- `src/build_all.py` is a driver that only calls the figure modules; not ported or copied.
