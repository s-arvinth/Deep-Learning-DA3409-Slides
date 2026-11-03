# Class 23 — Convolution Filters: figure code

Decks: `Class23_Convolution_Filters/A_Textbook_Figures/main.tex` and
`Class23_Convolution_Filters/B_Recreated_Figures/main.tex`. All code-generated
figures are loaded by the B deck only; the A deck uses only book crops.

Source: `B_Recreated_Figures/src/` (`common/style.py`, `common/models.py`,
`figures/fig04`-`fig11`). `common.py` here merges style and models; the
figure code, data, parameters and seeds (`default_rng(1)` in weight_matrix,
`default_rng(2)` in resolution, `default_rng(23)` in dense_vs_conv) are
unchanged. `fig10_rf_spotlight.py` wrote three files and was split into
`rf_spot_ab.py`, `rf_spot_c.py` and `rf_spot_d.py`.

Verification: every script was run in a fresh `python3` process from a clean
`output/`, and its PDF compared with the deck's existing figure by rendering
both with `pdftoppm -r 60 -png` and taking the mean absolute pixel difference
(0-255) and the share of pixels differing by more than 32. Environment:
python3, numpy, matplotlib (sci_style not installed, so the fallback
rcParams from CONVENTIONS.md were used, as for the original build); fonts from
the repository's `fonts/`.

| figure | script | slide(s) | mean diff | >32 share | status |
|---|---|---|---|---|---|
| conv_knobs | conv_knobs.py | "One dimension, one kernel" | 0.000 | 0.000% | match |
| dense_vs_conv | dense_vs_conv.py | "Images as Data" | 0.000 | 0.000% | match |
| kernels_2d | kernels_2d.py | "What a $3 \times 3$ kernel does" | 0.000 | 0.000% | match |
| receptive_field | receptive_field.py | "The recursion, checked" | 0.000 | 0.000% | match |
| resolution | resolution.py | "Resolution, measured" | 0.000 | 0.000% | match |
| resolution_circles | resolution_circles.py | "Resolution, drawn" | 0.000 | 0.000% | match |
| rf_spot_ab | rf_spot_ab.py | "How far back a unit can see, drawn" | 0.000 | 0.000% | match |
| rf_spot_c | rf_spot_c.py | "How far back a unit can see, drawn" | 0.000 | 0.000% | match |
| rf_spot_d | rf_spot_d.py | "How far back a unit can see, drawn" | 0.000 | 0.000% | match |
| weight_matrix | weight_matrix.py | "A convolution is a dense layer with most of it deleted" | 0.000 | 0.000% | match |

All 10 scripts: 10 match, 0 near, 0 MISMATCH (pixel-identical at 60 dpi).

## Figures with no script

| figure | deck | why |
|---|---|---|
| Bishop10_2, Bishop10_4 | A (Bishop10_4 also B) | book crop (`figs_official/`, copied unchanged into B `figs/`) |
| P_Conv1a, P_Conv2a, P_ConvDown, P_ConvUp, P_ConvNetworkRF | A | book crop (`figs_official/`) |
| P_ConvEquiInv | A, B | book crop (`figs_official/`, copied unchanged into B `figs/`) |
| `figs/Deep Learning.JPG`, `figs/3D_color.JPG` | B | external images, loaded by path |
| `figs/P_Conv1.pdf`, `figs/subsampled_textures.png` | — | present in B `figs/` but not loaded by either deck |

No source found: none. TikZ: none loaded via `\includegraphics`.

## Unused original modules

None: every module in `src/figures/` writes a figure loaded by the B deck.
(`src/build_all.py` is only a driver and is replaced by running the scripts.)
