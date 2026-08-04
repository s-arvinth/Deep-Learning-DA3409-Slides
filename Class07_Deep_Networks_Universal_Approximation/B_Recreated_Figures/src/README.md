# Figure sources — Class 7 (Deep Networks, Universal Approximation)

```bash
python3 build_all.py                   # all 9 figures -> ../figs/*.pdf
python3 build_all.py sawtooth          # only figures matching "sawtooth"
python3 figures/fig05_sawtooth.py      # or run one module directly
```

| module | output | what it shows |
|---|---|---|
| `fig01_compose` | `compose` | `f1`, `f2`, and the composition `f2∘f1` |
| `fig02_fold` | `fold` | the three branches of `f1`, each carrying a copy of `f2` |
| `fig03_compose2d` | `compose2d` | convex regions, then subdivision by a second layer |
| `fig04_buildup` | `buildup` | the two-layer computation stage by stage |
| `fig05_sawtooth` | `sawtooth` | `T`, `T∘T`, `T∘T∘T∘T`, and pieces vs depth (Telgarsky) |
| `fig06_ua_ridge` | `ua_ridge` | the ridge-function proof of universal approximation |
| `fig07_regions` | `regions` | linear regions bought per parameter, and per layer |
| `fig08_representation` | `representation` | half-moons: the learned representation |
| `fig09_hierarchy` | `hierarchy` | parts → motif → modulated motif |

`Bishop6_8.pdf` in `../figs/` is **not** generated here: it is
reproduced from Bishop & Bishop (2024), Fig. 6.8, and cited on the
slide that uses it.

Shared code lives in `common/`: `style.py` (palette, type sizes, `save`),
`data.py` (two moons, the 2-D target), `models.py` (the two running
networks, the 2-D network, the tent map `T` and its iterates, and the
parameter/region counts) — all written in the same notation as the
slides. See `../../../_shared/FIGURE_SOURCES.md`.

## The two new figures

`fig05_sawtooth` is the witness for the depth–width separation:
`T(x) = 2h(x) − 4h(x − 1/2)` is a two-unit ReLU network, and composing
it `L` times gives exactly `2^L` linear pieces from `2L` units. Panel
(d) contrasts that with the `2L + 1` pieces the same units buy in a
single layer.

`fig06_ua_ridge` illustrates the proof of universal approximation in `D`
dimensions: a target is approximated by a sum of ridge functions
`cos(wᵀx + b)`, each of which varies along a single direction and is
therefore reducible to the one-dimensional case proved in Class 5.
