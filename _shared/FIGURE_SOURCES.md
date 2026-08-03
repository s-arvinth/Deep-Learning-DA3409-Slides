# How figure source code is organised

Every `B_Recreated_Figures` deck carries the code that produced its
figures, laid out the same way so a figure can be found, tweaked and
re-rendered without touching anything else.

```
Class0N_.../B_Recreated_Figures/
  main.tex                 the deck
  figs/                    the rendered vector PDFs (what main.tex uses)
  fonts/                   Fira Sans/Mono, needed by both LaTeX and matplotlib
  src/
    build_all.py           regenerate everything, or a subset
    common/
      style.py             rcParams, the shared palette, save()
      data.py              the datasets the deck uses
      models.py            the running models, in slide notation
    figures/
      fig01_<name>.py      one module per figure, standalone-runnable
      fig02_<name>.py
      ...
```

## Regenerating

```bash
cd Class05_Shallow_NN_Regression/B_Recreated_Figures/src

python3 build_all.py                 # every figure in the deck
python3 build_all.py regions         # only figures matching "regions"
python3 figures/fig14_regions.py     # or run one module directly
```

Each module exposes `NAME` (the output filename, minus `.pdf`) and
`build()`. `build()` calls `style.use()` itself, so running a module on
its own gives exactly the same result as running it through
`build_all.py`.

## Changing a figure

Tweak the one module and re-run it. Nothing else regenerates, so the
rest of the deck is untouched and the git diff stays small.

* **Colours** — use the names from `common.style`: `L1` indigo,
  `L2` turquoise, `OUTC` maroon, `ACC` amber, `GREY`. Sequential data
  uses `style.SEQ` (viridis), ordered families use `style.CAT` (plasma).
  Do not introduce new colours.
* **Type sizes** — live only in `common/style.py:RC`. They are large on
  purpose: figures are scaled down on the slide, so what looks oversized
  in the PDF reads correctly from the back of the room.
* **Aspect** — prefer square panels; `style.square(ax)` does it.
* **Output** — always vector PDF via `style.save(fig, NAME)`. Heavy
  contour fills and 3-D surfaces are rasterised individually at 400 dpi
  and everything else stays vector.
* **Notation** — figure labels follow `_shared/NOTATION.md`, same as
  the slides.

## Dependencies

NumPy, Matplotlib, and SciencePlots (imported as `sci_style`, which
applies the `science` preset). The Fira faces are loaded from the deck's
own `fonts/` directory, so no system font installation is needed.
