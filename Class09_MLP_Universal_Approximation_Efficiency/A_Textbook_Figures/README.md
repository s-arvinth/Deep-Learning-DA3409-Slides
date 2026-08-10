# Class 9 — textbook-figures variant

Figures reproduced with attribution:

* **Bishop & Bishop (2024)**, Figs. 6.2–6.5 (fixed basis functions and
  the curse of dimensionality), 6.9 (the multilayer network), 6.10 (one
  two-layer network fitted to four different targets).
* **Prince (2023)**, Fig. 4.6 (the `L`-layer network with its
  dimensions) and Fig. 4.9 (linear regions against parameters, for
  `D = 1` and `D = 10`). CC BY-NC-ND,
  <https://udlbook.github.io/udlbook/>.

Four slides plot **published rates** — Barron, DeVore et al.,
Telgarsky, Bartlett et al. — which have no counterpart in either
textbook. Those plots are generated, and their source is in
`../B_Recreated_Figures/src/`. They are identical in both variants.

```bash
xelatex main.tex && xelatex main.tex     # twice
```
