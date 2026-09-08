# Class 17 — figure sources

Every figure in the B deck is generated here. Nothing is drawn by hand and
nothing is traced from a book: each number on a slide was measured by the
code below.

```bash
cd src
python3 build_all.py                       # all figures -> ../figs/
python3 build_all.py fig04                 # only the ones matching "fig04"
python3 figures/fig04_double_descent.py    # or run one directly
```

Output is vector PDF in `../figs/`, sized and styled by `common/style.py`
so every deck in the course shares one palette and one set of type sizes.

## Layout

```
src/
    build_all.py          discovers figures/fig*.py and calls build()
    common/style.py       palette, type sizes, save(), square()
    common/models.py      the three data-generating processes
    figures/fig01 … fig10 one module per figure, each exposing
                          NAME and build()
```

## The three models in `common/models.py`

| Object | What it is | Used for |
|---|---|---|
| `truth`, `sample` | a smooth 1-D regression problem, $h(x) + \mathcal{N}(0,\sigma^2)$ | the anatomy of the error |
| `piecewise_fit` | least squares in a basis of `K` bumps | capacity as a dial |
| `bias_variance` | Monte-Carlo bias$^2$ and variance over `L` data sets | Bishop Eqs. 4.51–4.52 |
| `minnorm_risk` | ridgeless least squares on an isotropic design | double descent, derived |
| `minnorm_risk_theory` | the closed form of Hastie et al. (2022), Thm. 1 | the overlay curve |

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig01_error_sources` | `error_sources` | noise, bias and variance, one panel each, following the anatomy of Prince (2023), Fig. 8.5 |
| `fig02_capacity_fits` | `capacity_fits` | twenty fits at two capacities, with the two measured terms printed; construction of Bishop Fig. 4.7 and Prince Fig. 8.8 |
| `fig03_tradeoff_curve` | `tradeoff_curve` | bias$^2$, variance and their sum against capacity and against the penalty, with the directly measured excess risk overlaid as a check |
| `fig04_double_descent` | `double_descent` | a random-feature model: training error to zero at $p=N$, test error peaking there and descending below its classical best |
| `fig05_minnorm_theory` | `minnorm_theory` | measured risk against the closed form, and the norm of the fitted vector peaking at the same place |
| `fig07_sample_wise` | `sample_wise` | capacity fixed, sample size varied: the range in which more data is worse |
| `fig08_smooth_interpolants` | `smooth_interpolants` | nine points, a gap, and interpolants that get smoother — and smaller in norm — as capacity grows |
| `fig09_double_descent_regimes` | `double_descent_regimes` | the fig04 experiment re-drawn in the layout of Bishop Fig. 9.9: one axis, the critical band shaded, both minima labelled |
| `fig10_double_descent_noise` | `double_descent_noise` | the same experiment at four label-noise levels, following Prince Fig. 8.10: the peak grows from 8 to over 4,000 |

## Notes on the choices

**Why a random-feature model in fig04 and a linear one in fig05.** The
random-feature model reproduces what the books report for deep networks,
including the second minimum being lower than the first. The isotropic
linear model does not always do that, but it is the one whose risk has a
closed form, so it is the one that lets the peak be *derived* rather than
only observed. The deck uses fig04 to establish the phenomenon and fig05
to explain it.

**Why fig08 fixes the bump width.** If the basis functions narrow as `K`
grows, extra capacity buys sharper spikes and the interpolant gets rougher
— the opposite of the argument being made. Holding the width fixed means
that adding bumps adds candidate solutions and nothing else, which is what
adding hidden units does.

**Why the epoch-wise slide uses Bishop's published Fig. 9.10 in every
variant.** A faithful small-scale reproduction of epoch-wise double descent
needs a regime this deck's models do not reach: gradient descent on a
fixed linear-in-features model can only approach the interpolation
threshold from below (its effective capacity is capped at $N$ by the rank
of the Hessian), so the risk along the path descends once and then rises,
with no second descent. Two-scale constructions in the manner of Heckel &
Yilmaz (2021) were tried and did not produce it either. Rather than invent
a curve that merely resembles the published one, the deck follows its own
rule for Bishop's Fig. 7.1 in Class 13: where the published figure is
genuinely better than anything that can be built honestly, use it in every
variant, with its citation.
