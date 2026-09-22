# Class 29 — figure sources

Every measured figure comes from experiments run here, on the reverse-mode
autodiff tape written for this block (the sandbox has no PyTorch): a
character-level LSTM language model on a short public-domain story, two
taggers on a small grammar, and three sequence classifiers on a signal
task. Nothing is loaded from a library, nothing is traced from a book, and
every number on a slide is printed by the code below.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig05               # only matching modules
python3 figures/fig06_pooling.py         # or one directly
```

The first run trains every model (about four minutes) and caches them
stage by stage in `../figs/_cache.pkl`, so an interrupted run resumes;
delete the cache to retrain.

## What `common/` provides

| Module | What it is |
|---|---|
| `tape.py` | the minimal reverse-mode tape of Classes 27–28: tensors, matmul, tanh, sigmoid, softmax, cross-entropy, concat, stack, Adam with norm clipping; checked against finite differences on import |
| `models.py` | `RNNCell` and `LSTMCell` (Class 28's cells); `run_layers` for stacking and reversal; `CharLM` (Jurafsky Eqs. 14.4–14.6, tied by Eqs. 14.12–14.14 or untied), `perplexity` (Eq. 3.14), sampling with a temperature (Eq. 7.51); `Tagger` (Fig. 14.7, left-to-right or bidirectional by Eqs. 14.16–14.18); `Classifier` (Fig. 14.8) with the last-state, mean (Eq. 14.15) and max read-outs; the grammar and the signal task |
| `experiment.py` | the shared experiments, one cached stage each: the tied LM, the untied twin, one/two/three layers at matched parameters, the two taggers, the three classifiers |
| `draw.py` | the unrolled-network layout helpers of Class 27, in the deck's colours |
| `style.py` | the deck's palette and figure sizes |

`data/magi.txt` is O. Henry, "The Gift of the Magi" (1905), public domain,
11,136 characters after whitespace normalisation; the last 15% is the
validation text.

## The figures

| Module | Figure | What is computed or drawn |
|---|---|---|
| `fig01_rnn_lm_training` | `rnn_lm_training` | the layout of Jurafsky Fig. 14.6: per-token losses over an unrolled LM |
| `fig02_char_lm` | `char_lm` | training and validation perplexity by step of the tied LSTM LM (d = 64, 700 steps); samples at temperatures 0.5, 1.0, 1.5 |
| `fig03_weight_tying` | `weight_tying` | embedding + output parameters against \|V\| at d = 512, tied and untied; the character model both ways: parameters and validation perplexity |
| `fig04_pos_tagging` | `pos_tagging` | the layout of Fig. 14.7 with the trained left-to-right tagger's distributions for "Janet will back the bill" |
| `fig05_bidirectional_labelling` | `bidirectional_labelling` | token accuracy of the two taggers, overall and on sentence-initial ambiguous words; their distributions for "back" in two sentences |
| `fig06_pooling` | `pooling` | ‖∂L/∂h_1‖ at initialisation against length for the three read-outs; training loss at n = 40 and 80 |
| `fig07_stacked_depth` | `stacked_depth` | validation perplexity of 1-, 2- and 3-layer LMs at about 38k parameters each |
| `fig08_four_architectures` | `four_architectures` | the layout of Jurafsky Fig. 14.15 |
| `fig09_stacked_bidirectional` | `stacked_bidirectional` | the layouts of Jurafsky Figs. 14.10 and 14.11 |

The two graphical-model drawings and the recursive tree are TikZ inside
`main.tex`.
