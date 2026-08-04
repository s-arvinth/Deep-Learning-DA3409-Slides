# Figure sources — Class 6 (Shallow NN: Classification)

```bash
python3 build_all.py                    # all 14 figures -> ../figs/*.pdf
python3 build_all.py softmax            # only figures matching "softmax"
python3 figures/fig10_softmax_posteriors.py   # or run one module directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_discriminant` | `discriminant` | `w` normal to the boundary, signed distance `a(x)/‖w‖` |
| `fig02_lsq_vs_logistic` | `lsq_vs_logistic` | least squares dragged by distant correct points |
| `fig03_gaussian_posterior` | `gaussian_posterior` | Gaussian class-conditionals; the posterior *is* a sigmoid |
| `fig04_gaussian_ml` | `gaussian_ml` | regression as maximum likelihood (the recipe, case 1) |
| `fig05_bernoulli` | `bernoulli` | the Bernoulli distribution for three values of `ŷ` |
| `fig06_sigmoid` | `sigmoid` | the logistic sigmoid and its derivative |
| `fig07_binary_model` | `binary_model` | `a(x)` → `σ(a)` → per-point likelihood |
| `fig08_saturation` | `saturation` | why squared error stalls and cross-entropy does not |
| `fig09_categorical` | `categorical` | softmax turning pre-activations into a categorical |
| `fig10_softmax_posteriors` | `softmax_posteriors` | three classes, posterior blend, entropy map |
| `fig11_multiclass_model` | `multiclass_model` | `K = 3` outputs, softmax, decision regions |
| `fig12_output_distributions` | `output_distributions` | one recipe, six output domains |
| `fig13_cross_entropy_kl` | `cross_entropy_kl` | negative log-likelihood as a KL divergence |
| `fig14_feature_space` | `feature_space` | XOR: input space, fixed features, learned `tanh` features |

Shared code lives in `common/`: `style.py` (palette, type sizes, `save`),
`data.py` (the datasets), `models.py` (sigmoid, softmax, logistic
regression, the shallow classifier, the Gaussian-posterior weights) —
all written in the same notation as the slides. See
`../../../_shared/FIGURE_SOURCES.md`.
