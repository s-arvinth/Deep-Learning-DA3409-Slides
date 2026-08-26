# Class 15 — textbook-figures variant

Reproduced with attribution:

* **Bishop & Bishop (2024)** — Fig. 7.7 (the effect of input data
  normalization), Fig. 7.8 (where batch and layer normalization take
  their averages).
* **Prince (2023)** — Fig. 7.7 (forward and backward variance over
  fifty layers at five initialization variances).
  CC BY-NC-ND, <https://udlbook.github.io/udlbook/>.

Eight points in this class have **no published figure** in either book:
the two ways the data alone can wreck the conditioning, the three
initialization rules compared, the failure of symmetric initialization,
the drift of the statistics during training, the effect of a normalizer
on the usable learning rate, the batch-size dependence of the
statistics, the smoothness of the error surface, and the scale
invariance of a normalized layer. This variant states them without a
figure. The **B variant** supplies purpose-built figures for all eight
and ships the code that makes them in `../B_Recreated_Figures/src/`.

Prince's Fig. 7.7 writes the initialization variance as `sigma^2_Omega`;
the caption gives the translation into the course notation.
