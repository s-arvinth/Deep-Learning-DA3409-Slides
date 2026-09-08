# Class 18 — figure sources

Every figure in the B deck is generated here; nothing is traced from a book
and every number printed on a slide was measured by this code.

```bash
cd src
python3 build_all.py                    # all figures -> ../figs/
python3 build_all.py fig05              # only matching modules
python3 figures/fig05_implicit_gd.py    # or one directly
```

## The three objects in `common/models.py`

| Object | What it is | Used for |
|---|---|---|
| `truth`, `sample`, `ridge_fit` | 1-D regression in a fixed bump basis | the penalty sweep |
| `quad_E`, `ridge_solution` | a 2-D quadratic error | weight decay in the eigenbasis |
| `rotated_hessian`, `ridge_solution_rot` | the same quadratic with its eigenvectors at an angle | the Fig. 9.3 geometry |
| `mm_model`, `mm_loss_grid`, `local_minima` | $\hat y = \sin(w_1 x) + \tfrac12\sin(w_2 x)$, and a grid-plus-descent search for every local minimum | the Fig. 9.1 surfaces |
| `L`, `grad_L`, `hess_L` | a 2-D loss with a *curved valley* of global minima | implicit regularization |
| `grad_L_mod` | the exact gradient of $L + \tfrac{\alpha}{4}\lVert\nabla L\rVert^2$ | testing Prince Eq. 9.8 |
| `flow` | RK4 gradient flow | so the integrator is not what is being measured |
| `batch_variance` | the SGD term of Prince Eq. 9.9 | testing Prince Eq. 9.9 |

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig01_underdetermined` | `underdetermined` | five exact fits to the same eight points, then a penalty choosing one |
| `fig02_weight_decay_geometry` | `weight_decay_geometry`, `shrinkage_factors` | the layout of Bishop Fig. 9.3 — error, penalty and sum contours with the eigen-axes drawn at $\mathbf{w}^\star$ — and, separately, each direction's shrinkage factor |
| `fig03_lasso_vs_ridge` | `qnorm_contours`, `lasso_vs_ridge` | the layouts of Bishop Figs. 9.5 and 9.6: nested $q$-norm level sets, and contact at a vertex versus on an arc with the constraint written inside the region |
| `fig04_penalty_sweep` | `penalty_sweep` | the same model at three $\lambda$, with the weight norm and the error against $h$ printed |
| `fig05_implicit_gd` | `implicit_gd` | the layout of Prince Fig. 9.3: loss, regularization term and their sum, with flow and descent on the first and modified flow and descent on the third |
| `fig06_implicit_sgd` | `implicit_sgd` | the layout of Prince Fig. 9.4: loss, gradient-descent term, stochastic term and their sum, as four surfaces |
| `fig07_loss_regularization` | `loss_regularization` | the layout of Prince Fig. 9.1: a two-parameter loss with nine minima, the penalty, and their sum with six |

## Notes on the choices

**Why the valley is curved.** A straight valley would let discrete descent
and continuous flow reach the same point, and there would be nothing to
show. Curvature is what makes the discretisation drift along the valley,
which is precisely the effect Prince Eq. 9.8 predicts.

**Why fig05 checks the order before drawing.** Panel (c) shows the modified
flow landing near the discrete endpoint, which could be luck. So the build
first measures, for seven step sizes, the distance from each flow's endpoint
to the discrete one, fits the log-log slopes over $\alpha \le 0.08$, and
asserts that the plain flow's error falls like $\alpha^{1.0}$ and the modified
flow's like $\alpha^{1.9}$ — the order the derivation claims, and not
something a wrong formula would reproduce. The slopes are printed and quoted
in the caption.

**Why the two figures with Bishop's layout use a rotated Hessian.** Bishop
draws the error's eigenvectors at an angle to the parameter axes, so that
"the flat direction" and "the $w_1$ axis" are visibly different things. The
geometry figure builds $\mathbf{H} = \mathbf{R}\,\mathrm{diag}(\lambda)\,\mathbf{R}^{\mathsf T}$,
solves $(\mathbf{H} + \alpha\mathbf{I})^{-1}\mathbf{H}\mathbf{w}^\star$ directly,
and asserts that this equals the eigenbasis shrinkage of Theorem 3. The lasso
figure searches for a configuration in which the lasso contact is exactly at
a vertex while the ridge contact keeps $\hat w_1 > 0$; both contact points are
found numerically on the boundary and printed.

**Why fig07 counts minima numerically.** Every local minimum is a grid cell
lower than its eight neighbours, refined by gradient descent and
de-duplicated; the counts in the panel titles are what that search found,
and the build asserts the sum has fewer than the loss. The global minimum
moves from $(3.05, 7.04)$ to $(2.80, 6.35)$.

**Why the bump width is fixed in fig01.** Narrowing the basis functions as
`K` grows would change what a single basis function can do; holding the
width fixed means extra capacity adds candidate solutions and nothing else.
