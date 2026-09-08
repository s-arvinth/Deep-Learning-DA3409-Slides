# Classes 17, 18 and 19 — one chapter, three lectures

The course plan assigns these three classes overlapping sources:

| Class | Date | Topic | Sources (course plan) |
|---|---|---|---|
| 17 | Mon 24 Aug | Revisit Bias-Variance and Double Descent | Bishop Ch. 9, Prince Ch. 8 |
| 18 | Tue 25 Aug | Regularization: Explicit and Implicit Regularization | Prince Ch. 9, Bishop Ch. 9 |
| 19 | Fri 28 Aug | Regularization: Heuristic Methods, Early stopping, Drop outs, Model averaging, Applying Noise | Prince Ch. 9, Bishop Ch. 9 |

**Bishop Ch. 9 is shared by all three.** Prince splits cleanly: Ch. 8 belongs
to Class 17 alone, Ch. 9 to Classes 18 and 19. So the whole partition problem
is Bishop Ch. 9, plus one back-reference to Bishop §4.3, which is where the
bias–variance decomposition is actually derived and which Class 17 "revisits".

---

## The organising question

| Class | The question | The object |
|---|---|---|
| **17** | *Where does test error come from, and why does more capacity stop hurting?* | the **error** and its decomposition |
| **18** | *What do we add to the objective, and what does the algorithm add by itself?* | the **penalty**, explicit and implicit |
| **19** | *What do we do to the training procedure and the data?* | the **procedure** |

The line between 17 and 18 is the one the books themselves draw. Class 17 is
diagnostic: it measures and decomposes generalization error and takes the
model as given. Classes 18 and 19 are prescriptive: they change something to
improve it. Class 17 never proposes a fix; Classes 18 and 19 never re-derive
the decomposition.

---

## Section-by-section assignment

### Class 17 — Revisit Bias-Variance and Double Descent

Bishop (2024):

- **§4.3 The Bias–Variance Trade-off** — Eqs. 4.41–4.52; Figs. 4.7, 4.8. This
  is the "revisit": the decomposition is derived in Ch. 4, not Ch. 9, and
  Ch. 9 refers back to it.
- §4.2 insofar as it supplies the optimal predictor $h(\mathbf{x}) = \mathbb{E}[t|\mathbf{x}]$
  and Eqs. 4.39, 4.42 — one slide, because the decomposition needs it
- **§9.3 Learning Curves** — the opening paragraph only: learning curves as a
  diagnostic, and the idea that effective complexity grows during training
- **§9.3.2 Double descent** — Figs. 9.9, 9.10, 9.11 in full

Prince (2023): **the whole of Chapter 8.**

- §8.1 Training a simple model
- §8.2 Sources of error; §8.2.1 Noise, bias and variance; §8.2.2 Mathematical
  formulation of test error — Eqs. 8.1–8.7; Figs. 8.5, 8.6
- §8.3 Reducing error; §8.3.1 Reducing variance; §8.3.2 Reducing bias;
  §8.3.3 Bias-variance trade-off — Figs. 8.7, 8.8, 8.9
- §8.4 Double descent; §8.4.1 Explanation — Figs. 8.10, 8.11, 8.12
- §8.5 Choosing hyperparameters — the validation-set protocol
- §8.6 Summary

Publications supplying results neither book states formally:
Belkin, Hsu, Ma & Mandal (2019); Nakkiran et al. (2019, 2021);
Hastie, Montanari, Rosset & Tibshirani (2022); Zhang et al. (2017);
Yilmaz & Heckel (2022); Bartlett et al. (2020).

Because the exact asymptotic risk of the minimum-norm interpolator is in
neither textbook, Class 17 has a **C variant**. It removes the four slides
that rest on Hastie et al. (2022): the closed-form ridgeless risk, its
two-branch reading, the theory-against-measurement figure, and the
sample-wise curve derived from the same formula. The
effective-model-complexity definition is *kept*, because Bishop §9.3.2 states
it explicitly and attributes it to Nakkiran et al. (2019).

**Built.** Class 17 ships as A (44 pages), B (46 pages) and C (38 pages).
Its narrative order is: what we are measuring, the decomposition and its
proof, the classical trade-off read off that decomposition, the measurement
that contradicts it, the mechanism, and what changes in practice. Nothing in
it proposes a fix.

Explicitly **not** in Class 17, and where it goes instead:

| Deferred | To |
|---|---|
| Weight decay, $L^2$, and its probabilistic reading | 18 |
| Inductive bias as a *subject* — inverse problems, no free lunch, symmetry, equivariance (Bishop §9.1) | 18 |
| Implicit regularization in GD and SGD (Prince §9.2) | 18 |
| Parameter sharing, soft weight sharing, residual connections (Bishop §9.4, §9.5) | 18 |
| Early stopping **as a regularizer** (Bishop §9.3.1, Prince §9.3.1) | 19 |
| Model averaging, ensembling, dropout (Bishop §9.6, Prince §9.3.2–9.3.3) | 19 |
| Applying noise, augmentation, transfer learning (Prince §9.3.4–9.3.8) | 19 |

The one term Class 17 must borrow is **inductive bias**, because Prince
§8.4.1 uses it to explain the over-parameterized regime. Class 17 states what
it means in one line and takes the mechanism no further; Class 18 develops it.
Similarly, Class 17 shows *epoch-wise* double descent (Bishop Fig. 9.10)
because it is a double-descent phenomenon, but says nothing about stopping
early as a method — that is Class 19.

### Class 18 — Regularization: Explicit and Implicit

Bishop: §9.1 Inductive Bias, with §9.1.1 Inverse problems, §9.1.2 No free
lunch theorem, §9.1.3 Symmetry and invariance, §9.1.4 Equivariance;
§9.2 Weight Decay, with §9.2.1 Consistent regularizers and §9.2.2 Generalized
weight decay; §9.4 Parameter Sharing and §9.4.1 Soft weight sharing;
§9.5 Residual Connections.

Prince: §9.1 Explicit regularization, §9.1.1 Probabilistic interpretation,
§9.1.2 $L^2$ regularization; §9.2 Implicit regularization, §9.2.1 in gradient
descent, §9.2.2 in stochastic gradient descent.

Publications supplying results neither book states formally:
Wolpert (1996); Tibshirani (1996); Nowlan & Hinton (1992); Simard et al.
(1992); Barrett & Dherin (2021); Smith et al. (2021); He et al. (2015);
Balduzzi et al. (2017); Bronstein et al. (2021).

Only one result used here is in neither set text: the eigenbasis shrinkage
factor $\lambda_i/(\lambda_i+\alpha)$ and the effective-parameter count that
follows from it, which Bishop §9.2 describes qualitatively and attributes to
Bishop (2006) and Hastie et al. (2009). The **C variant** removes it, together
with its proof slide and the shrinkage-factor figure; the Fig. 9.3 geometry
survives with a caption confined to what the published figure shows.

**Built.** Class 18 ships as A (50 pages), B (52 pages) and C (45 pages).
Its narrative order is: why a preference is unavoidable, the penalty and what
it does to each eigen-direction, the shape of the penalty, the preference the
optimiser already has, then bias built into the architecture. Nothing in it
touches the data or the stopping rule.

### Class 19 — Regularization: Heuristic Methods

Bishop: §9.3.1 Early stopping (including the weight-decay equivalence,
Fig. 9.8); §9.6 Model Averaging and §9.6.1 Dropout.

Prince: §9.3 Heuristics to improve performance — §9.3.1 Early stopping,
§9.3.2 Ensembling, §9.3.3 Dropout, §9.3.4 Applying noise, §9.3.5 Bayesian
inference, §9.3.6 Transfer and multi-task learning, §9.3.7 Self-supervised
learning, §9.3.8 Augmentation.

Publications supplying results neither book states formally:
Bishop (1995a, 1995b); Breiman (1996); Freund & Schapire (1996);
Srivastava et al. (2014); Gal & Ghahramani (2016); Szegedy et al. (2016);
Goodfellow et al. (2015).

The **C variant** removes the explicit filter form of the early-stopping /
weight-decay correspondence — Bishop §9.3.1 states the correspondence and
leaves the algebra to Exercise 9.6 and Bishop (1995a), and its ridge half
was already removed from the Class 18 C variant. The correspondence itself
survives, restated as Bishop gives it.

**Built.** Class 19 ships as A (43 pages), B (46 pages) and C (40 pages).
Its narrative order is: stop the run, average many runs, average implicitly
with dropout, perturb the inputs, then borrow other data. Every claim is
measured, because every method in the class is a heuristic.

---

## Cross-references to honour

- Class 13 introduced the training objective $E(\mathbf{w})$ and treated it as given.
  Class 17 is the first time the course asks what the *test* error is made of.
- Class 15 already used the word *normalization* for conditioning, not for
  generalization. Class 17 must not blur the two.
- Bishop's early-stopping/weight-decay equivalence (Fig. 9.8) rests on the
  Hessian eigen-decomposition from Class 13. Class 19 may assume it.
- Classes 9 and 10 proved depth–width separation results about
  *representational* capacity. Class 17 is about *statistical* capacity and
  should say so once, so the two notions are not confused.

## Notation

The frozen course notation (`_shared/NOTATION.md`) applies. Prince's
$\phi$ for parameters and $f[\mathbf{x},\phi]$ for the model become $\mathbf{w}$ and
$\hat y(\mathbf{x}; \mathbf{w})$; Bishop's $t$ for the target becomes $Y$, and his
$h(\mathbf{x})$ for the regression function is kept, since Prince's $\mu[\mathbf{x}]$
is the same object. Both books' data set $\mathcal{D}$ is kept.
