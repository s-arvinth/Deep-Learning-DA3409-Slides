# Class 23 — figure sources

Nothing in this class is statistical, so nothing here is random: every
figure is an exact computation on an explicit array, and the numbers on
the slides come from this code.

```bash
cd src
python3 build_all.py                       # all figures -> ../figs/
python3 build_all.py fig05                 # only matching modules
python3 figures/fig05_receptive_field.py   # or one directly
```

## What `common/models.py` provides

| Object | What it is |
|---|---|
| `conv1d` | the 1-D operation with stride, dilation and zero padding |
| `out_size` | the output-length formula |
| `conv_matrix` | the same layer written out as a matrix |
| `rf_size`, `measured_rf` | the receptive-field recursion, and the field measured by propagation |
| `conv2d`, `synthetic_image`, `KERNELS` | the 2-D operation and the constructed scene |
| `maxpool1d`, `upsample1d` | the two changes of resolution |
| `fc_params`, `conv_params` | exact parameter counts |

## Three self-checks

These are the reason the deck can state the formulas as fact:

1. `out_size` agrees with the length `conv1d` actually returns, over
   every combination of $n \in \{10,17,32\}$, $k \in \{1,3,5\}$,
   $s \in \{1,2,3\}$, $d \in \{1,2\}$, $p \in \{0,1,2\}$.
2. `rf_size` agrees with `measured_rf` at every depth, for every
   schedule used in the deck — `fig05` asserts this at build time, so
   the figure cannot be produced if the recursion is wrong.
3. `conv_matrix(n, w, s, d, p) @ x` reproduces `conv1d(x, w, s, d, p)`
   to machine precision.

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig09_knobs_prince` | `conv_knobs` | the layout of Prince Fig. 10.3: inputs in a column, weights in boxes, one shaded output per panel, for stride two, kernel five and dilation two; the inputs read are taken from `conv_matrix` |
| `fig04_weight_matrix` | `weight_matrix` | the layer as a matrix: dense, banded, and banded with stride |
| `fig05_receptive_field` | `receptive_field` | the span reaching one unit, and the recursion against the measurement |
| `fig06_kernels_2d` | `kernels_2d` | a constructed scene under blur, two edge kernels and a sharpener |
| `fig07_resolution` | `resolution` | two ways down, three ways back up, and how much shift pooling absorbs |
| `fig08_dense_vs_conv` | `dense_vs_conv` | the layout of Prince Fig. 10.4: six panels — the bipartite graph and its weight matrix for a dense layer, a kernel of three, and the same with stride two, every edge in the colour of its matrix entry |
| `fig10_rf_spotlight` | `rf_spot_ab`, `rf_spot_c`, `rf_spot_d` | the layout of Prince Fig. 10.6: layers as columns of units with a grey spotlight showing what reaches what; every field is computed by propagation and asserted equal to the recursion |
| `fig11_resolution_circles` | `resolution_circles` | the layout of Prince Figs. 10.11–10.12: a 4 × 4 array of numbers in tinted blocks, pooled three ways and restored three ways, every number computed |

## Notes on the choices

**Why the scene in fig06 is constructed.** A photograph would have to be
licensed and could not be regenerated. A disc, a bar, a strip and a
gradient are enough to show what an edge kernel responds to, and every
pixel comes from six lines of `models.py`.

**Why fig07(c) sweeps three window sizes.** With a window of two the
effect is invisible: pooled and unpooled sensitivities differ by about
8%. The claim is that a *wider* window absorbs more shift, so the figure
shows windows 2, 4 and 8 against the raw signal, normalised so the four
curves are comparable. At a shift of one the raw signal changes by 14%
and the eight-wide pooled signal by 6%.

**Why the three Prince layouts are redrawn rather than pasted.** Figs. 10.3,
10.4, 10.6 and 10.11–10.12 of Prince are diagrams of a calculation, and the
calculation is one this code already does. Redrawing them from `conv_matrix`,
from a propagated receptive field and from the pooling functions means the
picture and the deck's formulas cannot disagree: the inputs each output reads,
the units each field covers and every number in the pooled arrays are
produced by the same functions the slides quote.
