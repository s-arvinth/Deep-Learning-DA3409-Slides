# Class 13 figure sources

Every figure in `../figs/` is produced by exactly one module in
`figures/`. Nothing is hand-drawn and nothing is copied from a book.

```
python3 build_all.py              # rebuild every figure
python3 build_all.py gabor        # only modules matching "gabor"
python3 figures/fig05_descent_lemma.py    # or run one directly
```

Output goes to `../figs/<NAME>.pdf` as vector PDF at 400 dpi for the
rasterised layers, with fonts embedded (`pdf.fonttype = 42`).

## Layout

```
src/
  build_all.py            discovers and runs every figures/fig*.py
  common/
    style.py              the shared palette, type sizes and helpers
                          (identical to _shared/deck_style.py)
    models.py             the error surfaces, the Gabor model, the
                          optimisers, and the mini-batch estimator
  figures/
    fig02 … fig16         one module per figure, each exposing
                          NAME and build()
```

## One published figure is kept

`figs/Bishop7_1.pdf` is Bishop & Bishop's own Fig. 7.1. Nothing built
here improved on it, so this variant uses the original, cited on the
slide. Everything else below is generated.

## The figures

| Module | `NAME` | What it shows | Follows |
|---|---|---|---|
| `fig02_stationary_types` | `stationary_types` | minimum, maximum, saddle, by Hessian sign pattern | B&B Eq. 7.11 |
| `fig03_quadratic_ellipses` | `quadratic_ellipses` | elliptical contours on the eigenvectors; the effect of a wider spectrum | B&B Fig. 7.2 |
| `fig04_steepest_direction` | `steepest_direction` | the Cauchy–Schwarz bound by direction, and on the contours | Lemma 2 of the deck |
| `fig05_descent_lemma` | `descent_lemma` | the guaranteed decrease against $\eta$; three step-size regimes | Lemma 3 of the deck |
| `fig06_linreg_descent` | `linreg_descent` | descent on a convex least-squares loss, in data and parameter space | Prince Fig. 6.1 |
| `fig07_gabor_landscape` | `gabor_landscape` | the non-convex Gabor loss; three starts, three destinations | Prince Figs. 6.4, 6.5a |
| `fig08_sgd_escape` | `sgd_escape` | full batch vs batch of 3; each batch's own surface | Prince Figs. 6.5, 6.6 |
| `fig09_batch_variance` | `batch_variance` | the sampling distribution of the estimate; $\sigma/\sqrt{B}$ | B&B Section 7.2.4 |
| `fig10_batch_tradeoff` | `batch_tradeoff` | noise ball vs cost, per iteration and per gradient evaluation | — |
| `fig11_gradient_payoff` | `gradient_payoff` | $O(W^3)$ vs $O(W^2)$, and information per evaluation | B&B Section 7.2.1 |
| `fig12_shuffling` | `shuffling` | what a consecutive block contains, and what it does to training | B&B Section 7.2.4 |
| `fig13_convex_vs_nonconvex` | `convex_vs_nonconvex` | the chord test, on a convex and a non-convex function | Prince Section 6.1.2 |
| `fig14_curvature` | `curvature` | the eigenvalue as the curvature of a slice; why the steep direction is the short axis | B&B Eq. 7.11 |
| `fig15_step_regimes_2d` | `step_regimes_2d` | the four step-size regimes on the contour map | the descent lemma |
| `fig16_step_regimes_3d` | `step_regimes_3d` | the same four regimes on the surface | the descent lemma |

## Notation

`common/models.py` translates both books into the frozen course
notation once, at the source, so no figure and no slide carries two
conventions:

| Book | Book's symbol | Ours |
|---|---|---|
| Prince | `phi` (parameters) | `w` |
| Prince | `alpha` (learning rate) | `eta` |
| Prince | `L[.]` (loss) | `E(.)` |
| Bishop | `w`, `eta`, `E` | unchanged |

`phi` is never a parameter in this course — it is reserved for a fixed
basis function.

## Layout note

Every two-panel figure is a wide row of square panels rather than a
stacked grid. On a 16:9 slide the binding constraint is width, so a
single row gives each panel roughly twice the area a 2×2 grid would.

## Dependencies

numpy and matplotlib only. `sci_style` supplies the SciencePlots
`science` preset; if SciencePlots is not installed it falls back to an
equivalent rcParams block, so the modules run either way.
