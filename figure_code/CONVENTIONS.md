# Conventions for figure_code/ (read with REQUIREMENTS.txt)

These settle the points REQUIREMENTS.txt leaves open. Every class folder follows them.

## Scope and what must not change

- Work only inside `figure_code/`. Do not edit, move or delete anything else in the
  repository: no `.tex`, no `figs/`, no `figs_official/`, no `src/`. The original
  `src/` trees stay in place as the historical record.
- One folder per deck number: `figure_code/main_classN/` with N unpadded
  (`main_class5`, `main_class26`). A class's A and B variants are one deck number;
  the folder covers every code-generated figure loaded by either variant.

## Which figures get a script

- A figure gets a script if code in the class's `src/` (or `make_figures.py`)
  writes it. Book crops (`figs_official/`, names like `J14_7`, `G10_13`, `B7_2`,
  `P10_3`), images taken from elsewhere, and TikZ drawn inside the `.tex` get none.
- If a figure in `figs/` is loaded by a deck and looks code-generated but no code
  in the repo writes it, do not invent one: list it in `NOTES.md` under
  "No source found".

## Script layout

- `<name>.py`, where `<name>` is the `\includegraphics` argument exactly.
- `python <name>.py` writes `output/<name>.pdf` (or `.png`, matching the file the
  deck loads) inside the class folder. `python <name>.py --out DIR` writes to DIR
  instead. `output/` is git-ignored.
- Shared code goes in `common.py`. If a class genuinely needs more (an autodiff tape,
  model definitions, a cached experiment), keep it to as few helper modules as
  possible and name them for what they are (`tape.py`, `models.py`,
  `experiment.py`). Helpers are the only `.py` files whose names are not figure
  names.
- A script that runs a long experiment may cache its result in
  `output/_cache_<what>.pkl`, but it must recompute from scratch when the cache is
  absent, with fixed seeds, so that a clean run gives the same figure.

## Style and reproducibility

- Fonts: register the Fira `.ttf` files from the repository's root `fonts/` folder,
  located relative to `__file__` (`../../fonts` from a class folder).
- `sci_style`: `try: import sci_style` and on ImportError apply the fallback the
  figures were actually built with. That differs by deck, and each `common.py`
  documents which one it uses:
  - the plain block `plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white"})`
    (Classes 17–19, 23–29);
  - the plain block plus the SciencePlots "science" preset values (inward ticks on
    all four sides, minor ticks, `savefig.pad_inches` 0.05, mathtext tick labels),
    because those decks' figures were built with the real preset installed
    (Classes 2–15).
  The rule is to match the figure on the slide, verified with `tools/verify.py`.
- Every random seed is fixed inside the script or the helper it calls
  (`np.random.default_rng(<seed>)`, `np.random.seed`, `random.seed`), with the same
  values the original code used, so the figure matches the one on the slide.
- Save with `metadata={"CreationDate": None}` for PDFs so reruns are byte-stable
  where matplotlib allows it.

## Header comment (top of every figure script)

```python
"""<name> — <one line on what it shows>

Used in: Class N (<variant(s)>), slide(s): "<frame title>"[, "<frame title>"]
Needs:   numpy, matplotlib[, scipy, ...]  (optional: sci_style)
Run:     python <name>.py            -> output/<name>.pdf
"""
```

## Verification (required before a class counts as done)

1. Run every script in a fresh process, from a clean `output/` (delete caches first
   unless the experiment exceeds the time limit; then say so).
2. Compare each output with the deck's existing figure: render both at 60 dpi with
   `pdftoppm -r 60 -png` and compute the mean absolute pixel difference (0–255
   scale) and the share of pixels that differ by more than 32.
3. Write `NOTES.md` in the class folder with a table
   `| figure | script | slide(s) | mean diff | >32 share | status |`, status being
   `match` (mean < 1 and share < 0.5%), `near` (visually identical, small
   anti-aliasing or layout drift; say why) or `MISMATCH` (fix it, or explain).
   Also list: figures with no script and why (book crop, TikZ, no source found),
   and any original module moved to `figure_code/unused/main_classN/`.

## Unused code

- A module in the class's `src/` that writes no figure loaded by any deck is
  copied (not moved) to `figure_code/unused/main_classN/` with its original name.
