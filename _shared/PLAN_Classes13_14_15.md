# Classes 13, 14 and 15 — one chapter, three lectures

Classes 13, 14 and 15 are assigned the **same sources** in the course plan:

| Class | Date | Topic | Sources |
|---|---|---|---|
| 13 | Mon 17 Aug | Gradient Descent and optimization | Bishop Ch. 7, Prince Ch. 6–7 |
| 14 | Tue 18 Aug | Convergence | Bishop Ch. 7, Prince Ch. 6–7 |
| 15 | Wed 19 Aug | Normalization | Bishop Ch. 7, Prince Ch. 6–7 |

Because the three lectures draw on one chapter, the split has to be decided
**before** any slide is written, or the same material will be taught three
times. This file is that decision. Every subsection of the shared sources is
assigned to exactly one class. Nothing appears twice.

---

## The organising question

Each lecture answers one question, and only that question.

| Class | The question | The object being studied |
|---|---|---|
| **13** | *What is the step, and is it even a descent step?* | the **direction** and the **estimate** of it |
| **14** | *How fast does it get there, and under what conditions?* | the **dynamics** of the iteration |
| **15** | *How do we make the landscape and the signals well-scaled?* | the **conditioning** of the problem |

The dividing line between 13 and 14 is the one Bishop himself draws: §7.1.1
sets up the local quadratic model as **geometry** (what a minimum looks like);
§7.3 re-uses that same model as **dynamics** (how the iterate moves on it).
Class 13 takes the geometry, Class 14 takes the dynamics.

The dividing line between 14 and 15 is *what gets modified*. Class 14 modifies
the **iteration** (momentum, schedules, adaptive scaling) and leaves the
objective alone. Class 15 modifies the **problem** (where `w` starts, how the
activations are scaled) and leaves the iteration alone.

---

## Section-by-section assignment

### Class 13 — Gradient Descent and Optimization

Bishop (2024):

- §7.1 Error Surfaces — Eqs. 7.1, 7.2; stationary points; equivalent minima
- §7.1.1 Local quadratic approximation — Eqs. 7.3–7.14 (**geometry only**:
  Taylor model, Hessian eigen-decomposition, classification of stationary
  points, positive definiteness, the elliptical contours of Fig. 7.2)
- §7.2 Gradient Descent Optimization — Eq. 7.15, the iterative scheme
- §7.2.1 Use of gradient information — the O(W³) vs O(W²) counting argument
- §7.2.2 Batch gradient descent — Eq. 7.16
- §7.2.3 Stochastic gradient descent — Eqs. 7.17, 7.18; Algorithm 7.1
- §7.2.4 Mini-batches — the σ/√N argument; Algorithm 7.2; shuffling

Prince (2023):

- §6.1 Gradient descent — Eqs. 6.1–6.3; line search
- §6.1.1 Linear regression example — Eqs. 6.4–6.7; Fig. 6.1
- §6.1.2 Gabor model example — Eq. 6.8, 6.9; Figs. 6.2, 6.3
- §6.1.3 Local minima and saddle points — Figs. 6.4, 6.5a
- §6.2 Stochastic gradient descent — Fig. 6.5b
- §6.2.1 Batches and epochs — Eq. 6.10; Fig. 6.6
- §6.2.2 Properties of stochastic gradient descent (the six properties)
- §7.1 Problem definitions (the training objective restated) — one slide only

Explicitly **not** in Class 13, and where it goes instead:

| Deferred | To |
|---|---|
| Convergence rate, `(1 − ηλ)^T`, `η < 2/λ_max`, condition number | 14 |
| Momentum, Nesterov, learning-rate schedules | 14 |
| AdaGrad, RMSProp, Adam | 14 |
| Saddle points as an *obstacle to convergence* (they appear in 13 only as a feature of the landscape) | 14 |
| Parameter initialization (Bishop §7.2.5, Prince §7.5) | 15 |
| Data, batch and layer normalization (Bishop §7.4) | 15 |

### Class 14 — Convergence

Bishop: §7.3 opening (Fig. 7.3, the valley), Eqs. 7.24–7.30; §7.3.1 Momentum;
§7.3.2 Learning rate schedule; §7.3.3 RMSProp and Adam.

Prince: §6.2's learning-rate-schedule paragraph; §6.3 Momentum; §6.3.1
Nesterov accelerated momentum; §6.4 Adam; §6.5 Training algorithm
hyperparameters; the "Convexity, minima, and saddle points" box.

Publications supplying the theorems that neither book states formally:
Robbins & Monro (1951); Polyak (1964); Nesterov (1983); Duchi et al. (2011);
Tieleman & Hinton (2012); Kingma & Ba (2015); Reddi et al. (2018);
Loshchilov & Hutter (2017, 2019); Dauphin et al. (2014).

Because several of these results sit outside both textbooks, Class 14 has
a **C variant** (textbook-only theorems), as Classes 9 and 10 do. The
five results it removes are the optimal fixed learning rate and its
factor $(\kappa-1)/(\kappa+1)$, the step count in terms of $\kappa$,
Polyak's heavy-ball rate, Nesterov's $O(1/T^2)$, and the Robbins–Monro
conditions. Everything that survives is in Bishop §7.3 or Prince
§6.3–6.5.

**Built.** Class 14 ships as A (57 pages), B (64 pages) and C (57 pages).
Its narrative order is: the best case (equal curvature, one step), then
ill-conditioning as the problem, then momentum, schedules and adaptive
rates as the successive fixes.

### Class 15 — Normalization

Bishop: §7.2.5 Parameter initialization (symmetry breaking; He initialization,
Eqs. 7.19–7.23); §7.4 Normalization; §7.4.1 Data normalization; §7.4.2 Batch
normalization; §7.4.3 Layer normalization.

Prince: §7.5 Parameter initialization (the forward and backward variance
analysis); §7.1 insofar as it defines the exploding/vanishing quantities.

Publications: Glorot & Bengio (2010); He et al. (2015); Ioffe & Szegedy
(2015); Ba, Kiros & Hinton (2016); Wu & He (2018); Santurkar et al. (2018);
Xiong et al. (2020).

**Built.** Class 15 ships as A (45 pages), B (55 pages) and C (45 pages).
Its narrative order is: normalize the inputs, defeated by depth;
initialization, defeated by training; batch normalization, defeated by a
small batch; layer normalization. The four results it states from the
literature rather than the books — the feature-scale law for κ, batch
normalization's scale invariance, the smoothness explanation, and the
pre-norm/post-norm placement — are the four the C variant removes.

Initialization is taught **with** normalization rather than with gradient
descent, even though Bishop files it under §7.2, because its content is a
variance-propagation argument — the same argument that motivates every
normalization layer. Splitting them would mean deriving the same variance
recursion twice. Class 13 therefore states the practical default in a single
line and points forward.

---

## Cross-references to honour

- Class 11 already covered backpropagation and automatic differentiation
  (Prince §7.2–7.4, Bishop Ch. 8). Classes 13–15 **assume** the gradient is
  available and never re-derive it.
- Class 11 already covered vanishing and exploding gradients as a property of
  activation functions. Class 15 revisits the same phenomenon as a property of
  *scale*, and must say so explicitly rather than repeat the derivation.
- Class 6 already derived the three standard error functions. Class 13 treats
  `E(w)` as given.

## Notation

Both books are translated into the frozen course notation (see
`_shared/NOTATION.md`): `w` for every learnable parameter, `X` features,
`Y` targets. Prince's `ϕ` for parameters, `α` for the learning rate and
`L[·]` for the loss become `w`, `η` and `E(·)`. This is flagged once, on the
notation slide of each deck.
