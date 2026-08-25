# Class 14 — textbook-figures variant

Reproduced with attribution:

* **Bishop & Bishop (2024)** — Fig. 7.3 (the long valley and the
  oscillation across it), Figs. 7.4 and 7.5 (the two momentum regimes),
  Fig. 7.6 (momentum on the valley).
* **Prince (2023)** — Fig. 6.7 (SGD with and without momentum),
  Fig. 6.8 (the geometry of the Nesterov step), Fig. 6.9 (why one
  learning rate cannot serve two curvatures, and what Adam does about
  it).
  CC BY-NC-ND, <https://udlbook.github.io/udlbook/>.

Seven points in this class have **no published figure** in either book:
the distinction between a linear and a sublinear rate, the decoupled
decay of the eigen-coefficients, the one-step solution when the
curvature is equal in every direction, the cost of the condition
number, the shapes of the learning-rate schedules, the noise floor of a
fixed learning rate, and the effect of Adam's bias correction. This
variant states them without a figure. The **B variant** supplies
purpose-built figures for all seven and ships the code that makes them
in `../B_Recreated_Figures/src/`.
