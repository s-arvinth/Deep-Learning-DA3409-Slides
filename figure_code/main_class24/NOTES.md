# Class 24 — Applications: Detection and Segmentation: figure code

Decks: `Class24_Applications_Detection_Segmentation/A_Textbook_Figures/main.tex`
and `.../B_Recreated_Figures/main.tex`. All code-generated figures are loaded by
the B deck only; the A deck uses only book crops.

Source: `B_Recreated_Figures/src/` (`common/style.py`, `common/models.py`,
`common/bishopdraw.py`, `figures/fig01`-`fig09`). `common.py` here merges the
three helper modules (scripts use it as both `S` and `D`); the figure code and
parameters are unchanged. The only random draw (`default_rng(0)` in
output_shapes) is kept as is. No module wrote more than one figure.

Verification: every script was run in a fresh `python3` process from a clean
`output/`, and its PDF compared with the deck's existing figure by rendering
both with `pdftoppm -r 60 -png` and taking the mean absolute pixel difference
(0-255) and the share of pixels differing by more than 32. Environment:
python3, numpy, matplotlib (sci_style not installed, so the fallback
rcParams from CONVENTIONS.md were used, as for the original build); fonts from
the repository's `fonts/`.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| iou | iou.py | "IoU, drawn and measured" | 0.000 | 0.000% | match |
| output_shapes | output_shapes.py | "What the output structure decides" | 0.000 | 0.000% | match |
| sliding_window | sliding_window.py | "The saving, drawn" | 0.000 | 0.000% | match |
| transpose_conv | transpose_conv.py | "Transpose convolution, drawn" | 0.000 | 0.000% | match |
| unpooling | unpooling.py | "Unpooling, drawn" | 0.000 | 0.000% | match |

All 5 scripts: 5 match, 0 near, 0 MISMATCH (pixel-identical at 60 dpi).

## Figures with no script

| figure | deck | why |
|---|---|---|
| Bishop10_19, Bishop10_24, Bishop10_25, Bishop10_26a, Bishop10_26b, Bishop10_27, Bishop10_31 | A, B | book crops (`figs_official/`; the B `figs/` copies are byte-identical) |
| Bishop10_20, Bishop10_22, Bishop10_23, Bishop10_28a, Bishop10_28b, Bishop10_29, Bishop10_30 | A | book crops (`figs_official/`) |
| P_ConvImageNet, P_ConvAlex, P_ConvVGG, P_ConvImageNetPerformance, P_ConvYOLO, P_ConvSemSeg, P_ConvSemSegResults | A, B | book crops (`figs_official/`; B copies byte-identical) |
| `figs/Image_Report.png` | B | external image, loaded by path |
| `figs/subsampled_textures.png` | — | present in `figs/` but not loaded by either deck |

No source found: none. TikZ: none loaded via `\includegraphics`.

## Unused original modules

Copied (not moved) to `figure_code/unused/main_class24/`:

- `fig08_downup.py` — writes `downup.pdf`, whose `\includegraphics{downup}` in
  the B deck is commented out.
