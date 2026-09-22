# Class 27 — A variant (textbook figures)

Every figure is the published one, cited on the slide. Neither book ships
figure source, so each is cropped from the book PDF as a vector page
region by `crop_figures.py` (page located by caption text, region by the
text layout; a few tops are set by hand in `OVERRIDE`). The mathematics is
identical to the B variant; the B slides whose figures are experiments
with no published counterpart (training against inference, the measured
bottleneck, the convex hull, the beam run on the trained model) are
omitted, and every recreated book layout is replaced by the book's own.

| Slide | Figure | Source |
|---|---|---|
| Generating, one token at a time | `J14_9` | Jurafsky & Martin (2026), Fig. 14.9 |
| Two ways to hand over a vector: as an extra input | `G10_9` | Goodfellow et al. (2016), Fig. 10.9 |
| Two ways to hand over a vector: as the first state | `G10_12` | Goodfellow Fig. 10.12 |
| The whole model on the example | `J14_17` | Jurafsky Fig. 14.17 |
| The encoder–decoder, unrolled | `J14_18` | Jurafsky Fig. 14.18 |
| Teacher forcing, on the example | `J14_19` | Jurafsky Fig. 14.19 |
| Everything through one vector | `J14_20` | Jurafsky Fig. 14.20 |
| Attention, drawn: a context per step | `J14_21` | Jurafsky Fig. 14.21 |
| Attention, drawn: one decoder step in full | `J14_22` | Jurafsky Fig. 14.22 |
| Greedy against beam, on a toy vocabulary | `J13_7` | Jurafsky Fig. 13.7 |

Unused crops kept in `figs_official/` for the other decks of the block:
`J14_5`, `J14_16`, `G10_4`, `G10_6`, `G10_10`.

`crop_figures.py` tightens every crop to the drawn content with an even
margin after locating it, so framed figures keep their frame and nothing
drawn is cut (the same script as Class 26's).

Compile with `xelatex main.tex`, run twice. 48 pages.
