# Class 14 figure sources

Every figure in `../figs/` is produced by exactly one module in
`figures/`. Nothing is hand-drawn and nothing is copied from a book.

```
python3 build_all.py               # rebuild every figure
python3 build_all.py momentum      # only modules matching "momentum"
python3 figures/fig07_momentum_paths.py    # or run one directly
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
    models.py             the quadratic bowl, every optimiser, the
                          learning-rate schedules, and the optimal
                          constants used by the comparison figures
  figures/
    fig01 … fig15         one module per figure, each exposing
                          NAME and build()
```

## The figures

| Module | `NAME` | What it shows |
|---|---|---|
| `fig01_convergence_rates` | `convergence_rates` | linear against sublinear decay; the cost of a contraction factor near one |
| `fig02_eigen_decay` | `eigen_decay` | one $\eta$ acting on four curvatures; the stability constraint $\eta < 2/\lambda$ |
| `fig03_well_conditioned` | `well_conditioned` | $\kappa = 1$ solved in one step, and what a modest spread already costs |
| `fig04_condition_number` | `condition_number` | one contour at four condition numbers; steps to tolerance against $\kappa$ |
| `fig05_valley` | `valley` | the long valley, and three fixed learning rates on it |
| `fig06_momentum_regimes` | `momentum_regimes` | the displacements $\Delta \mathbf{w}^{(k)}$ under an error curve: aligned on low curvature, alternating on high. Follows the geometry of Bishop \& Bishop (2024), Figs. 7.4 and 7.5 |
| `fig07_momentum_paths` | `momentum_paths`, `nesterov_path` | descent, momentum and Nesterov on the same valley, **two panels per figure** so neither figure is wider than a slide |
| `fig08_lr_schedules` | `lr_schedules` | linear, power-law, exponential, cosine, warm-up + cosine |
| `fig09_noise_floor` | `noise_floor` | a fixed $\eta$ plateaus; a decaying one does not |
| `fig10_per_parameter` | `per_parameter` | the two bad choices of a single $\eta$, sign-only normalisation, and Adam |
| `fig11_optimiser_compare` | `optimiser_compare`, `optimiser_cost` | five update rules on one ill-conditioned quadratic: the two families on one figure, the error curves and the iteration counts on another |
| `fig12_bias_correction` | `bias_correction` | what $1/(1-\beta^{\tau})$ corrects, and how quickly it fades |
| `fig13_ellipse_geometry` | `ellipse_geometry` | the contour ellipse with its semi-axes named $\lambda^{-1/2}$, and why $-\nabla E$ misses $\mathbf{w}^\star$ when $\kappa > 1$ |
| `fig14_momentum_multiplier` | `momentum_multiplier` | the accumulated step length: $1/(1-\mu)$ when aligned, $1/(1+\mu)$ when alternating |
| `fig15_momentum_error` | `momentum_error` | the three runs by error, and $1-\rho$ against $\kappa$ for descent and momentum |

## The optimal constants

`common/models.py` exposes `gd_best`, `heavyball_best` and
`nesterov_best`, which return the step size, momentum and resulting
contraction factor that are optimal for a given pair of eigenvalues.
The comparison figures use them rather than hand-tuned numbers, so that
each method is shown at its best rather than at whatever happened to
look good.

## Notation

The optimiser signatures use the frozen course notation: `w`, `eta`,
`mu`, `E`. Prince's `phi`, `alpha`, `beta` and `L[.]` are translated
once, at the source, so no figure and no slide carries two conventions.

## Dependencies

numpy and matplotlib only. `sci_style` supplies the SciencePlots
`science` preset; if SciencePlots is not installed it falls back to an
equivalent rcParams block, so the modules run either way.
