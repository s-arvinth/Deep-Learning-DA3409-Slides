# Class 25 — Convolution Filters, Learned: figure code

Decks: `Class25_Convolution_Filters_Learned/A_Textbook_Figures/main.tex` and
`.../B_Recreated_Figures/main.tex`. All code-generated figures are loaded by the
B deck only; the A deck uses only book crops.

Source: `B_Recreated_Figures/src/` (`common/style.py`, `common/models.py`,
`figures/fig01`-`fig06`). `common.py` here merges style and models (including
the import-time self-check with its own `default_rng`); the figure code and
parameters are unchanged. No module wrote more than one figure.

Verification: every script was run in a fresh `python3` process from a clean
`output/`, and its PDF compared with the deck's existing figure by rendering
both with `pdftoppm -r 60 -png` and taking the mean absolute pixel difference
(0-255) and the share of pixels differing by more than 32. Environment:
python3, numpy, matplotlib (sci_style not installed, so the fallback
rcParams from CONVENTIONS.md were used, as for the original build); fonts from
the repository's `fonts/`.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| composition | composition.py | "Composition, computed" | 0.000 | 0.000% | match |
| receptive_field | receptive_field.py | "Reach grows with depth" | 0.000 | 0.000% | match |

All 2 scripts: 2 match, 0 near, 0 MISMATCH (pixel-identical at 60 dpi).

## Figures with no script

| figure | deck | why |
|---|---|---|
| Bishop10_9 ... Bishop10_18, Bishop10_32 | A | book crops (`figs_official/`) |
| Bishop10_10 | B | book crop (B `figs/` copy byte-identical to `figs_official/`) |

The B deck's lines loading `gabor`, `gradcam`, `fgsm`, Bishop10_12 ... Bishop10_18
and Bishop10_32 are commented out and were ignored.

No source found: none. TikZ: none loaded via `\includegraphics`.

## Unused original modules

Copied (not moved) to `figure_code/unused/main_class25/`:

- `fig04_gabor.py` — writes `gabor.pdf` (its `\includegraphics` is commented out)
- `fig05_gradcam.py` — writes `gradcam.pdf` (commented out)
- `fig06_fgsm.py` — writes `fgsm.pdf` (commented out)

The model functions they rely on (`gabor`, `ToyNet`, `grad_cam`, `make_shapes`,
`train_logistic`, `fgsm`, ...) remain in `common.py` because they were part of
the ported `models.py`.
