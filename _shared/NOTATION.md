# Notation contract — Introduction to Deep Learning

One symbol family, used in **every** deck, in **every** figure, and in
**every** caption. Based on Bishop & Bishop (2024). Where Prince (2023)
differs we keep Bishop's symbols and silently translate his figures.

## Bold means more than one number

Vectors and matrices are set in upright bold with `\mathbf{...}`.
Scalars, and the individual components of a vector or matrix, are left
in light italic. So `\mathbf{w}` is the parameter vector and `w_i` is
one weight; `\mathbf{H}` is the Hessian and `H_{ij}` is one entry.

Three things that stay light even though they look like they should not:

* `\nabla E` — the `\nabla` already marks it as a gradient, and this
  is Bishop's own convention
* a subscript that names a variance, as in `\sigma_w^2` — the `w` there
  is a label, not the vector
* any symbol that happens to be a scalar in this particular deck, such
  as `W` for the number of parameters, or the `s` of a power-law
  learning-rate schedule

Accents go outside the bold and are braced: `\hat{\mathbf{w}}`,
`\bar{\mathbf{a}}`. Never `\hat \mathbf{w}`.

## Data

| symbol | meaning |
|---|---|
| `x`, `\mathbf{x} \in \mathbb{R}^D` | input / **features** |
| `\mathbf{X}` | `N x D` data matrix |
| `y`, `\mathbf{y}` | **target**(s) |
| `\hat y(\mathbf{x},\mathbf{w})` | model **prediction** |
| `N` | number of examples |

## Sizes

| symbol | meaning |
|---|---|
| `D` | input dimension |
| `M` | hidden units — *and* number of basis functions (deliberate: they are the same role) |
| `K` | outputs |

## Parameters — always a `w`

| symbol | meaning |
|---|---|
| `\mathbf{w}`, `w_0` | weights, bias |
| `w^{(1)}_{ji}`, `w^{(1)}_{j0}` | input `i` -> hidden unit `j`, and its bias |
| `w^{(2)}_{kj}`, `w^{(2)}_{k0}` | hidden unit `j` -> output `k`, and its bias |
| `\mathbf{W}^{(1)}`, `\mathbf{W}^{(2)}` | the same in matrix form |

## Network internals

| symbol | meaning |
|---|---|
| `\phi_j(\mathbf{x})` | **fixed** basis function — *reserved*, never a parameter |
| `\boldsymbol{\Phi}` | design matrix, `\Phi_{nj} = \phi_j(\mathbf{x}_n)` |
| `a_j` | pre-activation |
| `h(\cdot)` | activation function |
| `z_j = h(a_j)` | hidden unit (activation) |
| `E(\mathbf{w})` | error function |

## The two rules we never break

1. `\phi` is **always** a fixed nonlinear feature map, **never** a
   parameter. (Prince uses `\phi` for output weights — we do not.)
2. Every learnable quantity is a `w`. (Prince's `\theta` for hidden-layer
   parameters becomes `w^{(1)}`.)

## Deviation from Bishop, noted once

Bishop writes targets as `t`. We write targets as `y` and predictions as
`\hat y`. This is the only symbol where we depart from the book, and each
deck states it on its notation slide.
