# figure_code/

One standalone Python script per code-generated figure in the DA3409 decks, named
exactly as the figure is loaded by `\includegraphics`. Built to the rules in
`REQUIREMENTS.txt`; the decisions those rules leave open are in `CONVENTIONS.md`.

The decks, `figs/` folders and the original `src/` trees are unchanged. This folder
was added alongside them on the branch `exp/figure-code-reorg`.

## Layout

```
figure_code/
  REQUIREMENTS.txt        the brief
  CONVENTIONS.md          output location, helpers, style fallback, header format, verification
  main_class<N>/          one folder per deck number (Classes 2-29 that have decks)
    <name>.py             writes output/<name>.pdf; `--out DIR` writes elsewhere
    common.py             fonts, palette, style, save()
    tape.py, models.py,   only where the class needs them (RNN classes, models shared
    experiment.py, ...    across figures); helpers are the only non-figure .py files
    data/                 input data the scripts read (Class 29: magi.txt)
    NOTES.md              per-figure comparison with the deck's figure, figures with
                          no script and why
  unused/main_class<N>/   original modules whose figures no deck loads (copies)
  tools/
    check_structure.py    names, helpers, headers against the decks
    verify.py             fresh re-run of every script and pixel comparison
```

## Running

```bash
cd figure_code/main_class14
python nesterov_path.py                      # -> output/nesterov_path.pdf
python nesterov_path.py --out ../../Class14_Convergence/B_Recreated_Figures/figs
```

Needs Python 3 with numpy and matplotlib (some classes also scipy); Pillow and
poppler (`pdftoppm`) for `tools/verify.py`. Fonts load from the repository's root
`fonts/`. `sci_style` is used if installed, otherwise the fallback in each
`common.py`, which reproduces the slides' figures.

## Status (verified 2026-11-03, fresh run from empty output/ folders)

| Classes | Figure scripts | Match the slide figure |
|---|---|---|
| 2, 3, 5, 6 | 45 | 45 |
| 7, 9, 10, 11, 13 | 43 | 43 |
| 14, 15, 17, 18, 19 | 47 | 47 |
| 23, 24, 25, 26 | 32 | 31 |
| 27, 28, 29 | 44 | 44 |
| **Total** | **211** | **210** |

The one exception is Class 26 `cost`: it plots measured wall-clock timings, so a
new run draws slightly different curves (mean pixel difference 2.9). Its plotting
code matches exactly when given the original timings; see `main_class26/NOTES.md`.

Class 9's A deck holds older copies of `uat_rate` and `barron_vs_curse` (denser
ticks); the scripts reproduce the current B figures. Class 29's A variant was not
committed in the parent repository, so only its B figures are covered.

```bash
python3 figure_code/tools/check_structure.py     # from the repository root
python3 figure_code/tools/verify.py 14 26        # or: all  (RNN classes take ~30 s each)
```
