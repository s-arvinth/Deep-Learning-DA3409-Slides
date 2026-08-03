# Figure sources — Class 5 (Shallow NN: Regression)

```bash
python3 build_all.py                 # all 16 figures -> ../figs/*.pdf
python3 build_all.py regions         # only figures matching "regions"
python3 figures/fig14_regions.py     # or run one module directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_linear_fit` | `linear_fit` | least-squares line with residuals drawn |
| `fig02_loss_surface` | `loss_surface` | `E(w)` surface + gradient-descent path |
| `fig03_basis_functions` | `basis_functions` | polynomial / Gaussian / sigmoid families |
| `fig04_conditional_mean` | `conditional_mean` | Gaussian noise, `p(y|x_0)`, the conditional mean |
| `fig05_relu` | `relu` | the ReLU activation |
| `fig06_activations` | `activations` | six activations on common axes |
| `fig07_buildup` | `buildup` | the scalar network computed stage by stage |
| `fig08_family` | `family` | twelve random three-unit networks |
| `fig09_approximate` | `approximate` | `M = 2, 5, 20` fits + measured error rate |
| `fig10_ua_construction` | `ua_construction` | the constructive proof, panel per step |
| `fig11_two_outputs` | `two_outputs` | `K = 2` outputs sharing a hidden layer |
| `fig12_two_inputs` | `two_inputs` | each unit as a hinge over `(x_1, x_2)` |
| `fig13_buildup2d` | `buildup2d` | hinges -> convex regions -> folded surface |
| `fig14_regions` | `regions` | Zaslavsky bound; joints on a scalar input |
| `fig15_learned_basis` | `learned_basis` | hidden units adapting to four targets |
| `fig16_fits` | `fits` | trained network on two noisy datasets |

`Bishop4_3.pdf` in `../figs/` is **not** generated here: it is reproduced
from Bishop & Bishop (2024), Fig. 4.3, and cited as such on the slide.

Shared code lives in `common/`: `style.py` (palette, type sizes, `save`),
`data.py` (datasets), `models.py` (the running networks, written in the
same notation as the slides). See `../../../_shared/FIGURE_SOURCES.md`.
