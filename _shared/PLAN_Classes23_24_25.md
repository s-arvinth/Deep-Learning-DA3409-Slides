# Classes 23, 24 and 25 — one chapter pair, three lectures

The course plan assigns these three classes the same two sources:

| Class | Date | Topic (course plan) | Sources |
|---|---|---|---|
| 23 | Mon 7 Sep | Convolution Networks — **Convolution Filters** | Prince Ch. 10, Bishop Ch. 10 |
| 24 | Tue 8 Sep | Convolution Networks — **Applications: Object Detection, Classification, Segmentation** | Prince Ch. 10, Bishop Ch. 10 |
| 25 | Wed 9 Sep | Convolution Networks — **Convolution Filters** | Prince Ch. 10, Bishop Ch. 10 |

Two of the three carry the same title, and the applications class sits
**between** them. That ordering constrains the split more than the titles do:
whatever Class 24 needs must already have been taught in Class 23, because
Class 25 comes afterwards. So Class 23 has to be the complete account of the
*operation* — one dimension and two, channels, receptive fields, and changes
of resolution — and Class 25 becomes the second sense of "convolution
filter": the filter as something **learned**, stacked, and inspected.

---

## The organising question

| Class | The question | The object |
|---|---|---|
| **23** | *What is a convolution, and why is it the right layer for an image?* | the **operation** |
| **24** | *What do you build out of it?* | the **task**: a label, a box, a mask |
| **25** | *What do the filters actually learn, and how deep should the stack be?* | the **learned filter** |

Class 23 is about a layer and says nothing about any particular network.
Class 24 is about three output structures and treats the trunk as given.
Class 25 opens the trunk: how depth composes filters, which architectures
became standard, and what a trained filter responds to.

---

## Section-by-section assignment

### Class 23 — Convolution Filters: the operation

Bishop (2024):

- §10.1 Computer Vision; §10.1.1 Image data — the parameter argument, and
  why permuting pixels destroys an image
- §10.2 opening — hierarchy, locality, equivariance, invariance
- §10.2.1 Feature detectors — Eq. 10.1, Fig. 10.1, Exercise 10.1 (the patch
  that maximises the response)
- §10.2.2 Translation equivariance — Fig. 10.2
- §10.2.3 Padding; §10.2.4 Strided convolutions
- §10.2.5 Multi-dimensional convolutions
- §10.2.6 Pooling

Prince (2023):

- §10.1 Invariance and equivariance — Eqs. 10.1, 10.2; Fig. 10.1
- §10.2 Convolutional networks for 1D inputs, **in full** — Eqs. 10.3–10.5;
  Figs. 10.2–10.7. The 1-D operation, padding, stride/kernel/dilation,
  convolutional layers, channels, receptive fields, and the MNIST-1D
  parameter comparison (2,050 against 150,185)
- §10.3 Convolutional networks for 2D inputs — Eq. 10.6; Figs. 10.8–10.10
- §10.4 Downsampling and upsampling, with §10.4.1–10.4.3 —
  Figs. 10.11–10.13

Publications supplying results neither book states formally:
LeCun et al. (1989, 1998); Fukushima (1980); Yu & Koltun (2016) for
dilation; Dumoulin & Visin (2016) for the arithmetic of output sizes;
Luo et al. (2016) for the effective receptive field; Cohen & Welling (2016)
for the equivariance converse.

Because two results are stated in neither textbook, Class 23 has a
**C variant**. It removes the converse half of the equivariance theorem —
that *every* translation-equivariant linear map is a convolution — and the
closed-form receptive-field recursion, keeping the worked table that Prince
builds figure by figure.

**Built.** Class 23 ships as A (38 pages), B (43 pages) and C (43 pages).
Its narrative order is: why a dense layer is impossible, what invariance and
equivariance demand, the operation in one dimension and its three knobs,
channels and receptive fields, two dimensions, and changes of resolution.
Nothing in it names a network.

Explicitly **not** in Class 23, and where it goes instead:

| Deferred | To |
|---|---|
| Image classification as a task, ImageNet, top-5 error | 24 |
| Object detection: boxes, IoU, sliding windows, scales, non-max suppression, region CNNs | 24 |
| Semantic segmentation, fully convolutional networks, U-net | 24 |
| Multilayer convolutions as a *stack* (Bishop §10.2.7) | 25 |
| Named architectures: AlexNet, VGG, and their descendants (Bishop §10.2.8) | 25 |
| Visualising trained CNNs: visual cortex, saliency maps, adversarial attacks, synthetic images (Bishop §10.3) | 25 |
| Style transfer (Bishop §10.6) | 25 |

Class 23 does use *up-sampling* (Prince §10.4.2), because it is an operation
on feature maps of exactly the same kind as the others, and because Class 24
needs it for segmentation two days later.

### Class 24 — Applications

Bishop: §10.4 Object Detection in full — §10.4.1 Bounding boxes,
§10.4.2 Intersection-over-union, §10.4.3 Sliding windows, §10.4.4 Detection
across scales, §10.4.5 Non-max suppression, §10.4.6 Fast region CNNs;
§10.5 Image Segmentation in full — §10.5.1 Convolutional segmentation,
§10.5.2 Up-sampling *as used for segmentation*, §10.5.3 Fully convolutional
networks, §10.5.4 The U-net architecture.

Prince: §10.5 Applications — §10.5.1 Image classification, §10.5.2 Object
detection, §10.5.3 Semantic segmentation; Figs. 10.15–10.21.

The organising idea is that the three tasks differ only in what the network
must emit — one label, a set of boxes with scores, or one label per pixel —
and that each output structure forces a different treatment of resolution.

Publications supplying results neither book states formally: Krizhevsky et
al. (2012) and Simonyan & Zisserman (2015) for the architectures whose
numbers are quoted; Sermanet et al. (2013); Redmon et al. (2016); Girshick
(2015) and Ren et al. (2015); Long et al. (2015); Noh et al. (2015);
Badrinarayanan et al. (2015); Ronneberger et al. (2015); Dumoulin & Visin
(2016).

Two results are in neither textbook, so Class 24 has a **C variant**: the
properties of IoU beyond its range (equality to one exactly at coincidence,
invariance under common rescaling) and the rank bound showing a linear
decoder from the bottleneck cannot place every pixel. The C deck keeps
Bishop's own statements in their place.

**Built.** Class 24 ships as A (54 pages), B (55 pages) and C (52 pages).
Its narrative order is: the three output structures; classification and the
trunk that shrinks; detection — boxes, IoU, sliding windows as convolution,
scales, suppression, direct grid prediction, region proposals;
segmentation — a label per pixel, down and back up, unpooling, transpose
convolution, fully convolutional networks, the U-net. The trunk is treated
as given throughout; what it learns is Class 25.

### Class 25 — Convolution Filters: what they learn

Bishop: §10.2.7 Multilayer convolutions; §10.2.8 Example network
architectures; §10.3 Visualizing Trained CNNs in full — §10.3.1 Visual
cortex, §10.3.2, §10.3.3 Saliency maps, §10.3.4 Adversarial attacks,
§10.3.5 Synthetic images; §10.6 Style Transfer.

Prince: the architectural detail of §10.5.1 that concerns the *trunk* rather
than the task, and Fig. 10.14.

Publications supplying results neither book states formally: Hubel &
Wiesel (1959); Fukushima (1980); Zeiler & Fergus (2013); Yosinski et al.
(2015); Selvaraju et al. (2016); Goodfellow et al. (2014); Eykholt et al.
(2018); Mordvintsev et al. (2015); Gatys et al. (2015).

Three results are in neither textbook, so Class 25 has a **C variant**: the
exact composition of two linear kernels into one, the ε‖w‖₁ arithmetic
behind the fast gradient sign method, and the translation invariance of
the style matrix. The C deck keeps Bishop's own statements in their place.

**Built.** Class 25 ships as A (44 pages), B (48 pages) and C (44 pages).
Its narrative order is: stacking filters (parameters, reach, composition,
the 1×1 convolution); a standard architecture (VGG-16 layer by layer,
parameters against connections, ImageNet); what a filter responds to (the
visual cortex, Gabor filters, learned first layers, patches that excite
deeper units, synthetic images); reading and fooling a network (Grad-CAM,
the fast gradient sign method, DeepDream); style transfer. The recurring
tool is the gradient with respect to the input.

---

## Cross-references to honour

- Class 18 defined invariance and equivariance and listed four ways to obtain
  an invariance, one of which was "build it into the architecture". Class 23
  is that fourth route, carried out. It must say so and must not re-derive the
  definitions.
- Class 18 also covered parameter sharing as a hard constraint. A convolution
  is the canonical instance, and Class 23 should name the connection once.
- Class 11 covered backpropagation. Class 23 states that the backward pass of
  a convolution is itself a convolution and does not re-derive automatic
  differentiation.
- Class 19 covered augmentation as the data route to invariance. Class 23
  contrasts the two routes in one line: augmentation asks the network to
  learn the symmetry, convolution gives it away for free.

## Notation

The frozen course notation (`_shared/NOTATION.md`) applies. Prince's
$\omega$ for kernel weights becomes $\mathbf{w}$; his $\Omega$ for the
per-layer weight array stays $\bm{\Omega}$ since it is not a scalar.
Bishop's filter and Prince's kernel are the same object and the deck uses
**kernel** throughout, noting the synonym once. Spatial indices are $i$ and
$j$; channel indices are $c$ (input) and $c'$ (output); layer indices are
$l$, as in Class 13.
