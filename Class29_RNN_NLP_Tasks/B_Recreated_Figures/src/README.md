# Class 29 — figure sources

Every figure is drawn by the code below. The two measured ones come from
experiments run here, on the reverse-mode autodiff tape written for this
block (the sandbox has no PyTorch): a character-level LSTM language model
on a short public-domain story, and a left-to-right tagger on a small
grammar. The rest recreate the books' figures exactly in their layouts, in
the deck's palette, with `common/bookdraw.py`. Nothing is loaded from a
library, nothing is pasted from a book, and every number on a slide is
printed by the code.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig02               # only matching modules
python3 figures/fig04_pos_tagging.py     # or one directly
```

The first run trains the models (about four minutes; `experiment.py`
still holds the untied, stacked and classifier stages of the first build,
which no slide uses now) and caches them stage by stage in
`../figs/_cache.pkl`, so an interrupted run resumes; delete the cache to
retrain.

## What `common/` provides

| Module | What it is |
|---|---|
| `tape.py` | the minimal reverse-mode tape of Classes 27–28: tensors, matmul, tanh, sigmoid, softmax, cross-entropy, concat, stack, Adam with norm clipping; checked against finite differences on import |
| `models.py` | `RNNCell` and `LSTMCell` (Class 28's cells); `run_layers` for stacking and reversal; `CharLM` (Jurafsky Eqs. 14.4–14.6, tied by Eqs. 14.12–14.14 or untied), `perplexity` (Eq. 3.14), sampling with a temperature (Eq. 7.51); `Tagger` (Fig. 14.7, left-to-right or bidirectional by Eqs. 14.16–14.18); `Classifier` (Fig. 14.8) with the last-state, mean (Eq. 14.15) and max read-outs; the grammar and the signal task |
| `experiment.py` | the shared experiments, one cached stage each: the tied LM, the untied twin, one/two/three layers at matched parameters, the two taggers, the three classifiers |
| `bookdraw.py` | the band/hidden/embedding/softmax/arrow helpers that reproduce Jurafsky's figure layouts (Class 27's module) |
| `draw.py` | the older unrolled-network layout helpers of Class 27 |
| `style.py` | the deck's palette and figure sizes |

`data/magi.txt` is O. Henry, "The Gift of the Magi" (1905), public domain,
11,136 characters after whitespace normalisation; the last 15% is the
validation text.

## The figures

| Module | Figure(s) | What is computed or drawn |
|---|---|---|
| `fig01_rnn_lm_training` | `rnn_lm_training` | Jurafsky Fig. 14.6 recreated: the unrolled LM with the next-word loss at every step |
| `fig02_char_lm` | `char_lm_curve`, `char_lm_samples` | training and held-out perplexity by step of the tied LSTM LM (d = 64, 700 steps); samples at temperatures 0.5, 1.0, 1.5 |
| `fig04_pos_tagging` | `pos_tagging` | Jurafsky Fig. 14.7 recreated, with the trained left-to-right tagger's distributions for "Janet will back the bill" |
| `fig08_four_architectures` | `four_architectures`, `arch_labelling`, `arch_classification`, `arch_lm`, `arch_encdec` | Jurafsky Fig. 14.15 recreated, and each of its four panels alone |
| `fig09_stacked_bidirectional` | `stacked_bidirectional` | Jurafsky Figs. 14.10 and 14.11 recreated side by side |
| `fig10_generation` | `generation` | Jurafsky Fig. 14.9 recreated (Class 27's figure) |
| `fig11_seq_classification` | `seq_classification`, `bidir_classification` | Jurafsky Figs. 14.8 and 14.12 recreated |
| `fig12_deep_ways` | `deep_a`, `deep_b`, `deep_c` | Goodfellow Fig. 10.13 recreated, one panel per file |
| `fig13_graphical` | `gm_complete`, `gm_state` | Goodfellow Figs. 10.7 and 10.8 recreated, with the dotted future nodes |
| `fig14_recursive` | `recursive_tree` | Goodfellow Fig. 10.14 recreated, every arrow labelled U, V or W |
