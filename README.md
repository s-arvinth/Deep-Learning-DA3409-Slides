# Fundamentals of Deep Learning — Lecture Slides

A collaborative Overleaf project holding the Beamer lecture decks for
*Fundamentals of Deep Learning*, all sharing one custom theme.

Decks included so far:

| Folder | Class | Topic |
|---|---|---|
| `Class02_Intro_Neural_Networks/` | 2 | Biology, McCulloch-Pitts neuron, linear separability |
| `Class03_Perceptrons_PLA/` | 3 | Perceptrons and the Perceptron Learning Algorithm |
| `Class05_Shallow_NN_Regression/` | 5 | Shallow networks: regression |
| `Class06_Shallow_NN_Classification/` | 6 | Shallow networks: classification |
| `Class07_Deep_Networks_Universal_Approximation/` | 7 | Deep networks, universal approximation |

---

## How to compile a deck (IMPORTANT - read this first)

Each class is a **separate, self-contained document**. To build one:

1. Open the class folder and click its `main.tex`.
2. In the **Menu** (top-left), set:
   - **Compiler -> XeLaTeX**  (required - the theme uses the Fira Sans font)
   - **Main document ->** the `main.tex` you want to compile
     (e.g. `Class05_Shallow_NN_Regression/main.tex`)
3. Click **Recompile**.

You switch which deck you're working on by changing the **Main document**
setting. Only one deck compiles at a time - that is normal.

> **Why XeLaTeX?** The theme is built on `metropolis`, which uses the
> **Fira Sans** typeface. The theme loads the Fira font files that are
> bundled with the project (there is a small `fonts/` folder inside each
> class folder, plus a full copy at the project root), so the correct font
> is used no matter what fonts Overleaf itself has installed. This only
> works under **XeLaTeX** (or LuaLaTeX) - if a deck renders in a plain
> serif/sans fallback, the compiler is not set to XeLaTeX.

---

## Project structure

```
/ (root)
  README.md
  beamerthememetropoliscustom.sty        <- the theme (shared by all decks)
  beamercolorthememetropoliscustom.sty   <- maroon / indigo palette
  beamerouterthememetropoliscustom.sty   <- footer bar, gradient title pages
  beamerinnerthememetropoliscustom.sty   <- bullets, blocks, design helpers
  fonts/                                  <- full Fira Sans / Fira Mono set
  Class02_.../  main.tex + figs_official/ + fonts/
  Class03_.../  main.tex + figs_official/ + fonts/
  Class05_.../  ...
  Class06_.../  ...
  Class07_.../  ...
```

> **About the `fonts/` folders.** Overleaf compiles a deck from *inside*
> its class folder, so each class folder carries the six Fira weights the
> theme needs. If you add a new class, copy the `fonts/` folder into it as
> well (step 3 below).

The four `beamer...metropoliscustom.sty` files live at the **project root**.
Overleaf searches the root for style files, so every deck finds the theme
automatically - you do not copy them into each folder.

Each deck's figures live in its own `figs_official/` subfolder. The figures
are the official vector PDFs from the textbooks' own repositories (Prince,
*Understanding Deep Learning*; Bishop & Bishop, *Deep Learning*) and from
Rojas, *Neural Networks*, reproduced with attribution on each slide.

---

## The shared theme

All decks use `\usetheme{metropoliscustom}`. Its ingredients:

- **Colours** - maroon (#800000) accents, an indigo/maroon gradient title
  page, and a yellow footer progress bar. Defined in
  `beamercolorthememetropoliscustom.sty`.
- **Design helpers** (in `beamerinnerthememetropoliscustom.sty`):
  `\redhrule`, `\redvrule`, the `redhighlight` box for the one key
  takeaway per slide, `\focus{n}{...}` / `\focusbullet{n}` for step-by-step
  reveals, and metropolis blocks.
- Each deck also defines a small **parallel-teaching strip** macro near the
  top of its `main.tex` (`\mlparallel`, `\biomodel`, `\mppar`, or `\shpar`)
  - a two-cell box drawing the parallel between two ideas.

Editing the theme once (at the root) changes every deck.

---

## Adding a new class

1. Duplicate an existing class folder (e.g. copy
   `Class05_Shallow_NN_Regression/` to `Class08_Your_Topic/`).
2. Rename it and edit its `main.tex`:
   - Update `\title`, `\subtitle`, `\author`, `\date`.
   - **Fix the graphics path** at the top of `main.tex` to point at the new
     folder name, e.g.
     `\graphicspath{{Class08_Your_Topic/figs_official/}{figs_official/}}`
3. Put that deck's figures in `Class08_Your_Topic/figs_official/`, and
   make sure the `fonts/` folder is present in it too (duplicating an
   existing class folder already brings it along).
4. Set it as the **Main document** and compile with **XeLaTeX**.

---

## Collaborating

- **Share** (top-right in Overleaf) -> invite collaborators by email, or
  turn on **Link sharing** so anyone with the link can edit.
- Everyone edits live; changes sync automatically.
- **History** (Menu -> History) tracks every change and lets you roll back.
- Convention: one deck per folder, the theme at the root, and each deck's
  figures in its own `figs_official/`.
