# Figure sources — Class 11 (Activations, Errors, Backprop, AD)

```bash
python3 build_all.py                     # all 9 figures -> ../figs/*.pdf
python3 figures/fig02_activation_derivatives.py    # or one directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_activation_zoo` | `activation_zoo_a`, `_b` | eight activations, four per figure in a wide row |
| `fig02_activation_derivatives` | `activation_derivatives_a`, `_b` | the same eight differentiated, each with `max h'` marked |
| `fig03_saturation_bound` | `saturation_bound` | `σ' ≤ 1/4`, `tanh' ≤ 1`, and the product over depth |
| `fig04_dying_relu` | `dying_relu` | a permanently dead unit, and how bias drift produces them |
| `fig05_vanishing_exploding` | `vanishing_exploding` | backpropagated norm per layer, tanh vs ReLU, three gains |
| `fig06_diff_methods` | `diff_methods` | finite-difference error vs step; `O(W²)` vs `O(W)` |
| `fig07_ad_modes` | `ad_modes` | forward vs reverse sweeps against `D` and `K` (variant A only; B uses a TikZ flowchart) |
| `fig08_heteroscedastic` | `heteroscedastic` | predicting `σ(x)` with a second output head |
| `fig09_mixture_density` | `mixture_density` | a multi-valued inverse problem and a mixture output |

`common/models.py` holds the eight activations with their analytic
derivatives, a deep chain for measuring how gradients travel
(`layer_gradient_norms`), and the dead-unit count (`dead_fraction`).

## Layout note

The activation figures are **one wide row of four square panels**, not a
2x4 or 2x2 grid. On a 16:9 slide the binding constraint is width, so a
single row lets each panel be roughly two and a half times the area it
had in the 2x4 version.
