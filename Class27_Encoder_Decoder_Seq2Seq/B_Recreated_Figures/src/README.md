# Class 27 — figure sources

Every measured figure comes from one experiment run here, on a
reverse-mode autodiff tape written for this block (the sandbox has no
PyTorch): two RNN encoder–decoders of the same size trained the same way on
a reversal task, one with the fixed context c = hᵉₙ and one with dot-product
attention. Nothing is loaded from a library, nothing is traced from a book,
and every number on a slide is printed by the code below.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig03               # only matching modules
python3 figures/fig03_bottleneck.py      # or one directly
```

The first run trains both models (about a minute) and caches them in
`../figs/_cache.pkl`; delete the cache to retrain.

## What `common/` provides

| Module | What it is |
|---|---|
| `tape.py` | a minimal reverse-mode tape: tensors, matmul, tanh, softmax, cross-entropy, concat, Adam with clipping; checked against finite differences on import |
| `models.py` | `Seq2Seq` in Jurafsky's notation (Eqs. 14.32–14.33, 14.35–14.37), the reversal task, teacher-forced training, greedy and beam decoding |
| `experiment.py` | the shared experiment: both models, 4,000 steps, lengths 2–10, tested to 16 |
| `draw.py`, `bookdraw.py` | layout helpers for the unrolled networks and the book's figure style, in the deck's colours |

## The figures

| Module | Figure | What is computed or drawn |
|---|---|---|
| `fig02_teacher_forcing` | `teacher_forcing` | the trained model at inference (own outputs fed back) and in training (gold inputs, per-token losses) on one source |
| `fig03_bottleneck` | `bottleneck` | training loss and exact-match accuracy by source length for the fixed-context and the attention model |
| `fig05_context_hull` | `context_hull` | encoder states of one source in two principal directions, the contexts and the fixed vertex |
| `fig06_beam_search` | `beam_toy`, `beam_search` | Jurafsky Fig. 13.7's tree with the book's probabilities, greedy against beam; the same tree on the trained model, every branch the model's probability |
| `fig07_generation` | `generation` | the layout of Jurafsky Fig. 14.9 |
| `fig08_conditioning` | `condition_input`, `condition_encdec` | the layouts of Goodfellow Figs. 10.9 and 10.12 |
| `fig09_seq2seq_book` | `seq2seq_full`, `seq2seq_unrolled` | the layouts of Jurafsky Figs. 14.17 and 14.18 on the example sentence |
| `fig10_teacher_forcing_book` | `teacher_forcing_book` | the layout of Jurafsky Fig. 14.19 on the example sentence |
| `fig11_bottleneck_book` | `bottleneck_book` | the layout of Jurafsky Fig. 14.20, and the same condensed to two blocks |
| `fig12_attention_book` | `attention_top`, `attention_full` | the layouts of Jurafsky Figs. 14.21 and 14.22 |

`common/bookdraw.py` holds the layout helpers for the book's figures
(embedding columns, hidden boxes, softmax boxes, bands, braces).
