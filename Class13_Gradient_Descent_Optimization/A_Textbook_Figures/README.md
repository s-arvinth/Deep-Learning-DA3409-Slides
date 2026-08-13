# Class 13 — textbook-figures variant

Reproduced with attribution:

* **Bishop & Bishop (2024)** — Fig. 7.1 (the error surface over weight
  space), Fig. 7.2 (elliptical contours on the eigenvectors of the
  Hessian).
* **Prince (2023)** — Fig. 6.1 (gradient descent on the linear model),
  Fig. 6.2 (the Gabor model), Fig. 6.3 (its training set), Fig. 6.4 (a
  non-convex loss and its minima), Fig. 6.5 (gradient descent vs
  stochastic gradient descent), Fig. 6.6 (each batch's own surface), and
  the convexity panel of Section 6.1.2.
  CC BY-NC-ND, <https://udlbook.github.io/udlbook/>.

Eight points in this class have **no published figure** in either book:
the three Hessian sign patterns drawn as surfaces, the Cauchy–Schwarz
picture behind steepest descent, the eigenvalue read as the curvature of
a slice, the four step-size regimes on the contours and on the surface,
the sampling distribution of the mini-batch gradient, the batch-size
trade-off, and the effect of an unshuffled file. This variant states
them without a figure. The **B variant** supplies purpose-built figures
for all eight, and ships the code that makes them in
`../B_Recreated_Figures/src/`.

Where a figure uses Prince's notation on its axes, the caption gives the
translation into the course notation ($\phi \to w$, $\alpha \to \eta$,
$L[\cdot] \to E(\cdot)$).
