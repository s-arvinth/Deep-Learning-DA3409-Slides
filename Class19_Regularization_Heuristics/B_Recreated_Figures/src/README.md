# Class 19 — figure sources

Every method in this class is a heuristic, so every claim in the deck is
measured here rather than asserted.

```bash
cd src
python3 build_all.py                     # all figures -> ../figs/
python3 build_all.py fig02               # only matching modules
python3 figures/fig02_early_vs_decay.py  # or one directly
```

## The three settings in `common/models.py`

| Object | What it is | Used for |
|---|---|---|
| `truth`, `sample`, `features` | 1-D regression in a fixed bump basis | the object being stopped, averaged, thinned and perturbed |
| `sgd_path` | plain SGD, optionally with dropout on the features | learning curves, dropout |
| `gd_filter`, `ridge_filter`, `matched_alpha` | the two coefficient filters and Bishop's $\alpha = 1/\eta\tau$ | early stopping vs weight decay |
| `bagged_ensemble` | L models on L bootstrap resamples | committees |
| `train_with_input_noise`, `ridge_solution` | descent with fresh input noise, and its claimed equivalent | the noise theorem, checked |
| `sgd_input_noise` | the bump model trained by SGD with fresh noise on the *inputs* | the Prince Fig. 9.10 panels |
| `gd_path_quadratic` | the path descent takes from the origin on a quadratic | the Bishop Fig. 9.8 geometry |
| `bayes_posterior`, `bayes_predictive` | the exact Gaussian posterior and predictive in the bump basis | the Prince Fig. 9.11 panels |

## The figures

| Module | Output | What it shows |
|---|---|---|
| `fig01_learning_curves` | `learning_curves` | training and validation error for one run, and the fit before, at and after the minimum |
| `fig02_early_vs_decay` | `early_stopping_path`, `early_vs_decay` | the layout of Bishop Fig. 9.8 — error contours on the eigen-axes, the descent path, $\hat{\mathbf{w}}$ beside the matched weight-decay solution — on its own; then the two filters at three run lengths and the directions each leaves alive, as a pair |
| `fig03_ensemble_gain` | `ensemble_gain` | bagged members and their mean, and the committee error against $M$ beside the $1/M$ law |
| `fig04_input_noise` | `input_noise` | the layout of Prince Fig. 9.10: the bump model trained by SGD with input noise at three levels, ten samples of the perturbed data dotted; the exact linear-model equivalence is asserted in the build (under 2% at every level) rather than drawn |
| `fig05_dropout` | `dropout` | fits with and without dropout, and test error against the dropout rate |
| `fig06_bayesian` | `bayesian` | the layout of Prince Fig. 9.11: two posterior samples under three prior variances, and the predictive mean with two standard deviations, all from the exact Gaussian posterior |

## Notes on the choices

**Why fig02 counts directions.** Comparing two filters curve-by-curve
shows they have the same shape, which is suggestive. Summing them over a
spectrum gives a single number — how many parameters each method leaves
alive — and the two agree to within 7% of $W$ across four decades of run
length. That turns Bishop's qualitative sentence into a measurement.

**Why fig03 shows the law failing.** Bishop's Eq. 9.50 gives a factor of
$M$, under an independence assumption that bagged members do not satisfy.
Plotting the measured curve against $1/M$ shows a factor of 4.5 where 64
was promised, which is the honest version of the result and makes the
free bound $E_{\mathrm{COM}} \le E_{\mathrm{AV}}$ the one worth relying on.

**Why fig04(b) is drawn from the penalty side.** Panel (a) establishes
that input noise and weight decay give the same solution, exactly, for a
linear model. Once that is established, drawing the consequence from the
penalty side is the same picture with less Monte-Carlo noise in it.

**Why fig05(b) does not find 0.5.** It finds 0.1, on a model with 44
features. The usual 0.5 is calibrated on far wider networks with much
more redundancy. Reporting what was measured, rather than the familiar
default, is the point of the panel.
