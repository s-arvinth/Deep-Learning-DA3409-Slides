# Class 6 — recreated-figures variant (pending)

The folder structure matches every other class so the repository stays
uniform, but the figures here have **not been rebuilt yet**: `main.tex`
is currently a copy of `../A_Textbook_Figures/main.tex` and still points
at textbook images.

To bring it in line with Classes 2, 3, 5 and 7:

1. Add one module per figure under `src/figures/` (see
   `../../Class05_Shallow_NN_Regression/B_Recreated_Figures/src/` for the
   pattern — `common/style.py`, `common/data.py`, `common/models.py`).
2. Run `python3 src/build_all.py` to write vector PDFs into `figs/`.
3. Replace the `\includegraphics` targets in `main.tex` and rewrite the
   captions.

`src/common/style.py` and `src/build_all.py` are already in place and are
identical to the Class 5 versions, so the palette and type sizes will
match automatically.
