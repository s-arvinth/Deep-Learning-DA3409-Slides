# Figure sources — Class 10 (Depth and Width Efficiency)

```bash
python3 build_all.py                       # all 5 figures -> ../figs/*.pdf
python3 figures/fig02_compositional.py     # or one module directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_depth_width` | `depth_width` | shallow fits to `T∘T∘T`, and the width needed |
| `fig02_compositional` | `compositional` | Poggio et al.: `(D−1)ε^(−2/m)` against `ε^(−D/m)` |
| `fig03_depth_width_asymmetry` | `depth_width_asymmetry` | exponential cost of flattening vs polynomial cost of narrowing |
| `fig04_regions_bounds` | `regions_bounds` | Montúfar lower and product upper bounds vs parameters |
| `fig05_vc_growth` | `vc_growth` | `VCdim = Θ(W L log W)` and the sample bound it implies |

`fig01`, `fig04` and `fig05` moved across from Class 9 when the two
lectures were rebalanced; `fig02` and `fig03` are new.

Constants in `fig03` panel (b) are illustrative — only the growth rates
(polynomial, and linear for ReLU) are claimed by the theorems.
