# Introduction to Deep Learning — lecture slides

Beamer decks for the course, built on the custom `metropoliscustom`
theme (Fira Sans, maroon accents, gradient title pages, yellow footer
progress bar). **Compile with XeLaTeX, twice.**

## Contributing

This project is edited from two places at once: directly in the Overleaf
web editor, and locally through this git clone. To keep both in step,
always sync through the helper script rather than raw `git`:

```bash
cd ~/Documents/TA-Duty

./overleaf-sync.sh pull                # before you start editing
./overleaf-sync.sh push "what changed" # commits, rebases on Overleaf, pushes
./overleaf-sync.sh whoami              # check who commits are attributed to
```

`push` always rebases on the latest Overleaf state before pushing, so
changes made in the web editor are never overwritten.

Commits are authored by whoever runs the script. Set your identity once
so the history is attributed to you:

```bash
git -C ~/Documents/TA-Duty/DeepLearning-Overleaf config user.name  "Your Name"
git -C ~/Documents/TA-Duty/DeepLearning-Overleaf config user.email "you@example.com"
```

To mirror to GitHub as well:

```bash
./overleaf-sync.sh github git@github.com:<you>/<repo>.git
```

After that, `push` sends to both remotes.

## Layout

Every class has **two self-contained variants**. Each compiles on its
own, and each carries its own `fonts/` and figures — Overleaf compiles
from inside the subfolder, so nothing may be shared across folders.

```
Class02_Intro_Neural_Networks/
  A_Textbook_Figures/            figures reproduced from the books, cited
  B_Recreated_Figures/           figures built for this course, with source
Class03_Perceptrons_PLA/         A_… / B_…
Class05_Shallow_NN_Regression/   A_… / B_…
Class06_Shallow_NN_Classification/ A_… / B_…
Class07_Deep_Networks_Universal_Approximation/ A_… / B_…
Class09_MLP_Universal_Approximation_Efficiency/ A_… / B_… / C_…
    C_Textbook_Theorems_Only/    B, minus every result absent from both books
Class10_Depth_Width_Efficiency_Theorems/       A_… / B_… / C_…
Class11_Activations_Errors_Backprop_AD/        A_… / B_…
Class13_Gradient_Descent_Optimization/         A_… / B_…
Class14_Convergence/                           A_… / B_… / C_…
Class15_Normalization/                         A_… / B_… / C_…

_shared/
  NOTATION.md                    the symbol contract every deck follows
  PLAN_Classes13_14_15.md        how Bishop Ch.7 and Prince Ch.6-7 are
                                 split across the three classes that
                                 share them, with no overlap
  FIGURE_SOURCES.md              how src/ is organised and how to re-render
  deck_style.py                  the shared matplotlib style
  fonts/                         master copy of the Fira faces

beamer*metropoliscustom.sty      the theme (repo root)
overleaf-sync.sh                 lives one level up, in ~/Documents/TA-Duty
```

## Figure sources

Each `B_Recreated_Figures` deck ships the code that produced its figures,
one module per figure:

```
B_Recreated_Figures/src/
  build_all.py            regenerate all, or a subset by name
  common/style.py         palette, type sizes, save()
  common/data.py          the datasets
  common/models.py        the running models, in slide notation
  figures/fig01_….py      one file per figure, standalone-runnable
```

See `_shared/FIGURE_SOURCES.md` for the full convention.

## Notation

All decks use a single symbol family, documented in
`_shared/NOTATION.md`: `x` features, `y` targets, `\hat y` predictions,
`w` for every learnable parameter, and `\phi_j` reserved exclusively for
fixed basis functions. Anything that departs from Bishop & Bishop (2024)
is stated on each deck's notation slide.

## Compiling locally

```bash
cd Class05_Shallow_NN_Regression/B_Recreated_Figures
TEXINPUTS="../..://:" xelatex main.tex && TEXINPUTS="../..://:" xelatex main.tex
```

On Overleaf, set the main document to the `main.tex` of the variant you
want and the compiler to XeLaTeX.
