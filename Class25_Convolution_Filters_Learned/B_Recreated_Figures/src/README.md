# Class 25 — figure sources

Every recreated figure is an exact computation or a small experiment run
by this code; nothing is traced from a book. The published results figures
(AlexNet's filters, the Zeiler–Fergus patches, the synthetic images, the
VGG saliency maps, the panda, the stop signs, DeepDream, style transfer)
are used as printed and cited, since they are the outputs of networks and
data this deck does not have.

```bash
cd src
python3 build_all.py                 # all figures -> ../figs/
python3 build_all.py fig06           # only matching modules
python3 figures/fig06_fgsm.py        # or one directly
```

## What `common/models.py` provides

| Object | What it is |
|---|---|
| `conv2d`, `relu` | the layer in two dimensions |
| `stack_params`, `stack_rf`, `compose` | parameters and reach of a stack of kernels, and the single kernel equal to two in succession |
| `vgg16_walk` | Bishop's Fig. 10.10 architecture with exact parameters and connections per layer |
| `gabor` | Bishop Eqs. 10.6–10.8 |
| `ToyNet`, `grad_cam` | a two-layer filter bank with a class score, its gradient written by hand, and Bishop Eqs. 10.9–10.10 |
| `make_shapes`, `train_logistic`, `fgsm`, `accuracy` | a two-class image problem, a linear classifier trained here, and Bishop Eq. 10.11 |
| `style_matrix` | Bishop Eq. 10.15 |
| `synthetic_image`, `KERNELS` | the scene and the four kernels of Class 23 |

## Self-checks in the build

1. `compose(W1, W2)` applied once equals `W1` then `W2` applied in turn, to
   machine precision — asserted when the models are imported and again in
   `fig02`.
2. `vgg16_walk` totals 138.4M parameters with 102.8M in the first fully
   connected layer, the figures Bishop quotes (138M, nearly 103M); the
   slide quotes them beside Bishop's Fig. 10.10.
3. `ToyNet.grad_wrt_layer1` is the exact gradient (a ReLU mask sent back
   through each second-layer kernel), so the Grad-CAM weights are exact
   averages, not estimates.
4. The logistic classifier in `fig06` reaches 99% held-out accuracy before
   the attack is run; the accuracies printed are measured after it.

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig01_receptive_field` | `receptive_field` | Bishop Fig. 10.9 redrawn in the deck's colours: three layers, kernel three, the units and edges one output depends on; the counts from `stack_rf` |
| `fig02_composition` | `composition` | an edge kernel then a blur, the 5×5 kernel equal to the pair, and how a ReLU between them breaks the equality |
| `fig04_gabor` | `gabor` | the layout of Bishop Fig. 10.11 from Eq. 10.6 |
| `fig05_gradcam` | `gradcam` | Grad-CAM on the toy network: the four maps with their α_k and the saliency map over the scene |
| `fig06_fgsm` | `fgsm` | one attack on a trained linear classifier, and accuracy against ε |

## Notes on the choices

**Why the attack is on a linear model.** Bishop's own point is that the
phenomenon is not over-fitting, since a much less flexible linear model
is fooled in the same way. A logistic classifier trained here shows that
directly, and the linear arithmetic (ε‖w‖₁) can be checked against what
happens. A deep network would have needed data this deck does not ship.

**Why Grad-CAM is on a two-layer toy.** The definition asks for a gradient
through the network, and the honest way to show one without an autodiff
library is a network small enough to differentiate on paper. The α_k and
the saliency map are exact for that network.

**Why VGG-16 is Bishop's own figure.** `vgg16_walk` still computes the
exact parameter and connection counts quoted on the slide, but the
architecture drawing is Bishop's Fig. 10.10 as printed.
