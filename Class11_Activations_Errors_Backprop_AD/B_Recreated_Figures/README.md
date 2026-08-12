# Class 11 — recreated-figures variant

First lecture on **how** to train, after six on **what** to build.
Covers activation functions (chosen for their derivative), error
functions beyond the three of Class 6, backpropagation with a full
derivation, and automatic differentiation.

Class 6 already derived the standard error functions from maximum
likelihood; that is recapped in one slide and then **extended** — the
canonical-link theorem, heteroscedastic outputs and mixture density
networks — not repeated.

```bash
cd src && python3 build_all.py
cd ..  && xelatex main.tex && xelatex main.tex     # twice
```
