# Class 15 figure sources

Every figure in `../figs/` is produced by exactly one module in
`figures/`. Nothing is hand-drawn and nothing is copied from a book.

```
python3 build_all.py             # rebuild every figure
python3 build_all.py variance    # only modules matching "variance"
python3 figures/fig03_variance_depth.py   # or run one directly
```

## Layout

```
src/
  build_all.py            discovers and runs every figures/fig*.py
  common/
    style.py              the shared palette, type sizes and helpers
                          (identical to _shared/deck_style.py)
    models.py             the least-squares Hessian, variance
                          propagation, the four normalizers and their
                          exact backward passes, and a small trainable
                          ReLU stack
  figures/
    fig01 … fig11         one module per figure, each exposing
                          NAME and build()
```

## The figures

| Module | `NAME` | What it shows |
|---|---|---|
| `fig01_input_scaling` | `input_scaling` | raw and rescaled data, and the two error surfaces they produce |
| `fig02_condition_from_data` | `condition_from_data` | unequal scale (fixable) against correlation (not fixable) |
| `fig03_variance_depth` | `variance_depth` | forward and backward variance over fifty layers at five initializations |
| `fig04_init_rules` | `init_rules` | He, Glorot and LeCun against fan-in; the distribution at layer fifty |
| `fig05_symmetry` | `symmetry` | identical units stay identical; random ones separate |
| `fig06_drift` | `drift` | the statistics moving away from where initialization put them |
| `fig07_norm_axes` | `norm_axes` | which entries are pooled by batch, layer, group and instance norm |
| `fig08_bn_effect` | `bn_effect` | three learning rates, with and without a normalizer |
| `fig09_bn_batchsize` | `bn_batchsize` | the standard error of the batch statistics against `B` |
| `fig10_smoothness` | `smoothness` | how much `E` varies over one step, and how far the gradient can be trusted |
| `fig11_scale_invariance` | `scale_invariance` | the output is unchanged under weight scaling; the gradient scales as `1/a` |

## Correct backward passes

`common/models.py` implements `batch_norm_backward` and
`layer_norm_backward` exactly rather than as a straight-through
approximation:

```
dE/da = ( B G - sum(G) - z * sum(G z) ) / (B sigma)
```

Every training-curve figure uses them, so the batch-normalized runs are
genuinely batch-normalized and the comparison is honest.

## One figure that is deliberately absent

There is no measured figure for pre-norm against post-norm. A faithful
demonstration needs a real transformer-scale stack; a two-layer toy
reproduces the *opposite* ordering and would mislead. The deck therefore
carries a TikZ diagram of the two placements and cites Xiong et al.
(2020) for the quantitative claim.

## Notation

Bishop writes `w` and `a`, Prince writes `Omega` and `f`. Both are
translated into the frozen course notation at the source, so no figure
and no slide carries two conventions.

## Dependencies

numpy and matplotlib only. `sci_style` supplies the SciencePlots
`science` preset, with an equivalent rcParams fallback if it is absent.
