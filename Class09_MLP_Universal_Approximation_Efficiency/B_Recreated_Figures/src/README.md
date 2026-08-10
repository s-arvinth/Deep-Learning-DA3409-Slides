# Figure sources — Class 9 (MLP, Universal Approximation, Efficiency)

```bash
python3 build_all.py                 # all 7 figures -> ../figs/*.pdf
python3 build_all.py curse           # only figures matching "curse"
python3 figures/fig05_depth_width.py # or run one module directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_curse_grid` | `curse_grid` | a grid in D = 1, 2, 3, and the count `S^D` |
| `fig02_curse_volume` | `curse_volume` | volume on the skin, Gaussian shell, data needed |
| `fig03_uat_rate` | `uat_rate` | best M-unit fits and the measured rate, against Barron |
| `fig04_barron_vs_curse` | `barron_vs_curse` | Barron's `M^(-1/2)` against the Sobolev `W^(-s/D)` bound |
| `fig05_depth_width` | `depth_width` | shallow fits to `T∘T∘T`, and the width needed |
| `fig06_regions_bounds` | `regions_bounds` | Montúfar lower and product upper bounds vs parameters |
| `fig07_vc_growth` | `vc_growth` | `VCdim = Θ(W L log W)` and the sample bound it implies |

Shared code is in `common/`: `style.py` (palette, type sizes, `save`),
`data.py` (targets), `models.py` (the tent map and its iterates,
parameter counts, region counts, the Barron and Sobolev rates, the VC
formula, and a best piecewise-linear fit) — all in slide notation.

## Where the numbers come from

Every formula plotted here is the statement of a theorem in the deck:

* regions, shallow — Zaslavsky (1975), `Σ_{j≤D} C(M,j)`
* regions, deep lower bound — Montúfar et al. (2014)
* regions, upper bound — the product bound of Serra et al. (2018)
* `M^(-1/2)` — Barron (1993)
* `W^(-s/D)` — DeVore, Howard & Micchelli (1989)
* `Θ(W L log W)` — Bartlett, Harvey, Liaw & Mehrabian (2019)

The `depth_width` figure uses the tent map `T(x) = 2h(x) − 4h(x − 1/2)`,
the witness in Telgarsky's (2016) depth separation.
