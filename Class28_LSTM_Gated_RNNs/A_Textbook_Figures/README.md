# Class 28 — A variant (textbook figures)

Every figure is the published one, cited on the slide. Neither book ships
figure source, so each is cropped from the book PDF as a vector page
region by `crop_figures.py` (page located by caption text, region by the
text layout; two tops are set by hand in `OVERRIDE`). The mathematics is
identical to the B variant; the three B slides whose figures are
experiments with no published counterpart (the gradient by lag, the gate
activity, the three cells on one task) are omitted, and the leaky-unit
slide keeps its proposition without the impulse-response plot.

| Slide | Figure | Source |
|---|---|---|
| The composition, drawn | `G10_15` | Goodfellow et al. (2016), Fig. 10.15 |
| The LSTM cell, drawn | `J14_13` | Jurafsky & Martin (2026), Fig. 14.13 |
| The LSTM cell, as a block diagram | `G10_16` | Goodfellow Fig. 10.16 |
| Gated units are drop-in modules | `J14_14` | Jurafsky Fig. 14.14 |
| Clipping at a cliff, drawn | `G10_17` | Goodfellow Fig. 10.17 |
| Where the gate leads: explicit memory | `G10_18` | Goodfellow Fig. 10.18 |

Compile with `xelatex main.tex`, run twice. 39 pages.
