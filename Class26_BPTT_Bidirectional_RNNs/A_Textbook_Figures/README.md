# Class 26 — A variant (textbook figures)

Every figure is the published one, cited on the slide. Neither book ships
figure source, so each is cropped from the book PDF as a vector page
region by `crop_figures.py` (page located by caption text, region by the
text layout; the tops the heuristic misses are set by hand in
`OVERRIDE`). The mathematics is identical to the B variant; the B slides
whose figures are experiments with no published counterpart (the BPTT
check, the per-step terms, the timings, the influence maps) are omitted.
The four introductory schematics of the B variant (space against time,
the four tasks, the window, the running example) recreate no published
figure and are kept, from `figs/`; where the books have a drawing of the
same idea (Jurafsky Figs. 14.5, 14.15) it is used instead.

| Slide | Figure | Source |
|---|---|---|
| What natural language processing asks of a model | `J14_15` | Jurafsky & Martin (2026), Fig. 14.15 |
| Why the tools we have fall short | `J14_5` | Jurafsky Fig. 14.5 |
| The simple recurrent network, drawn | `J14_1` | Jurafsky Fig. 14.1 |
| The same network, as one feedforward step | `J14_2` | Jurafsky Fig. 14.2 |
| Unfolding, drawn | `G10_3` | Goodfellow et al. (2016), Fig. 10.3 |
| Three design patterns | `G10_4`, `G10_5` | Goodfellow Figs. 10.4, 10.5 |
| Forward inference, and the unrolled network | `J14_3`, `J14_4` | Jurafsky Figs. 14.3, 14.4 |
| Both directions, drawn | `G10_11` | Goodfellow Fig. 10.11 |
| Both directions, as two networks | `J14_11` | Jurafsky Fig. 14.11 |

Unused crops kept in `figs_official/`: `G10_1`, `G10_2` (the dynamical
system and the network without outputs).

`crop_figures.py` now tightens every crop to the drawn content with an
even margin after locating it, so framed figures keep their frame and
nothing drawn is cut.

Compile with `xelatex main.tex`, run twice. 53 pages.
