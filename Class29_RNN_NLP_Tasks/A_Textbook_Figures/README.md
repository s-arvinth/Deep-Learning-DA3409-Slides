# Class 29 — A variant (textbook figures)

Every figure is the published one, cited on the slide. Neither book ships
figure source, so each is cropped from the book PDF as a vector page
region by `crop_figures.py` (page located by caption text, region by the
text layout; the tops that the layout heuristic misses are set by hand in
`OVERRIDE`). The mathematics is identical to the B variant; the five B
slides whose figures are experiments with no published counterpart (the
trained language model, weight tying tried, the three read-outs, stacking
tried, bidirectionality measured) are replaced by the book's drawing of
the same idea or omitted.

| Slide | Figure | Source |
|---|---|---|
| Training by self-supervision | `J14_6` | Jurafsky & Martin (2026), Fig. 14.6 |
| Generation, drawn | `J14_9` | Jurafsky Fig. 14.9 |
| Sequence labelling: a label at every step | `J14_7` | Jurafsky Fig. 14.7 |
| Sequence classification, drawn | `J14_8` | Jurafsky Fig. 14.8 |
| RNNs as modules | `J14_10`, `J14_11` | Jurafsky Figs. 14.10, 14.11 |
| Three ways to make an RNN deep | `G10_13` | Goodfellow et al. (2016), Fig. 10.13 |
| Bidirectional classification, drawn | `J14_12` | Jurafsky Fig. 14.12 |
| The two graphs, drawn | `G10_7`, `G10_8` | Goodfellow Figs. 10.7, 10.8 |
| The tree-shaped cousin | `G10_14` | Goodfellow Fig. 10.14 |
| Four architectures | `J14_15` | Jurafsky Fig. 14.15 |

Unused crops kept in `figs_official/`: `G10_3` (the unrolled RNN),
`G10_11` (the bidirectional RNN, Class 26's figure).

Compile with `xelatex main.tex`, run twice. 39 pages.
