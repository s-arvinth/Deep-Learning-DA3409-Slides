# Class 24 — figure sources

Nothing in this class is statistical, so nothing here is random: every
recreated figure is an exact computation on an explicit array or a published
architecture specification, and the numbers on the slides come from this
code. Photographs, results figures and schematic architecture drawings from the
books (ImageNet, YOLO, the Wayve images, the Noh et al. network and results,
Bishop's non-max suppression, segmentation-network and U-net figures) are
used as published and cited on the slide.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig04               # only matching modules
python3 figures/fig04_sliding_window.py  # or one directly
```

## What `common/models.py` provides

| Object | What it is |
|---|---|
| `iou`, `nms` | the box metric and Bishop's suppression rule, as functions |
| `window_cost` | multiply-adds for the toy sliding-window network, naive and shared |
| `conv2d_matrix`, `transpose_conv` | a strided convolution as a matrix, and the scattering up-sampling |
| `avgpool`, `maxpool`, `unpool_avg`, `unpool_max`, `bilinear_up` | the resolution changes |
| `vgg19_spec`, `hourglass_spec` | published architecture schedules, kept for reference |
| `bishopdraw` | sheared perspective grids and numbered flat grids in Bishop's four colours, for Figs. 10.22–10.23 and 10.28–10.30 |
| `synthetic_image` | the constructed scene of Class 23 |

## Self-checks in the build

1. `transpose_conv` equals `conv2d_matrix(...).T @ z` to machine precision
   (Bishop's Exercise 10.13, done numerically) — `fig07` asserts it.
2. `avgpool(unpool_avg(z)) == z` and `maxpool(unpool_max(z)) == z`
   (Proposition 8) — checked when the models are loaded.
3. `window_cost` asserts that the enlarged network's output count equals
   the number of window positions it claims to evaluate.
4. `vgg19_spec` totals 143.7M parameters, the figure Prince quotes as 144M.

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig01_output_shapes` | `output_shapes` | output values per image for the three tasks, and what each output looks like |
| `fig03_iou` | `iou` | the layout of Bishop Fig. 10.20, plus IoU against pixel error for a wide and a narrow box |
| `fig04_sliding_window` | `sliding_window` | Bishop Figs. 10.22 and 10.23 redrawn in perspective: the toy network on its 6×6 window and enlarged to 8×8, every map size from the layer arithmetic |
| `fig07_transpose_conv` | `transpose_conv` | Bishop Fig. 10.30 redrawn in perspective; the build asserts the scattering equals $\mathbf{M}^{\mathsf T}\mathbf{z}$ |
| `fig08_downup` | `downup` | the Class 23 scene pooled and restored one to four times, with the error |
| `fig09_unpooling` | `unpooling` | Bishop Figs. 10.28 and 10.29 redrawn: duplication, max-unpooling, and unpooling to the remembered positions, every number from the pooling functions |

## Notes on the choices

**Why four of Bishop's diagrams are redrawn rather than pasted.** Figs.
10.22–10.23 and 10.28–10.30 are diagrams of a calculation the code already
does; redrawing them from `window_cost`, the pooling functions and
`transpose_conv` means the picture and the deck's numbers cannot disagree.
The purely schematic figures (10.25, 10.27, 10.31) carry no calculation and
are used as printed.

**Why the sliding-window count uses stride two.** Bishop's toy network pools
non-overlapping 2×2 blocks, so the enlarged network of Fig. 10.23 evaluates
windows two pixels apart, four of them on an 8×8 image. The naive count
uses the same four positions; comparing against all nine stride-one
positions would flatter the shared network.

**Why the bottleneck figure uses average pooling and bilinear up-sampling.**
Both are fixed, parameter-free operations, so the error measured is what
the resolution change itself discards, with no training in the loop to
blame or credit.
