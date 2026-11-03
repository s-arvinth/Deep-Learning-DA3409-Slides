# Class 11 — Activations, Error Functions, Backpropagation, Automatic Differentiation

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 39 | 50 | 13 | 0 errors, 0 warnings | IDENTICAL (54 pages) |
| B_Recreated_Figures | 40 | 48 | 13 | 0 errors, 0 warnings | IDENTICAL (55 pages) |

Tags are the first lines of every frame body. (build_deck reports hbox=2 for A and hbox=1 for B;
the PDFs are identical to the baselines, so these overfull boxes predate the tagging.)

## \srcdef keys used (both variants)
baur1983, baydin2018 (both already in keys.md; text copied from it). Fixed keys: bishop2024, prince2023.

## How the numbers were checked
- Bishop (pdftotext of the course PDF): Fig 6.12 and Eqs 6.14-6.18 in 6.2.3; Eq 5.72 (sigma' = sigma(1-sigma));
  5.4.6 Canonical link functions, Eqs 5.89-5.95 (5.95 = gradient {y_n - t_n} phi_n);
  6.5 MDN: Fig 6.17 (forward/inverse toy data, x + 0.3 sin 2 pi x), Fig 6.18 (MDN), Eq 6.38, Fig 6.19;
  8.1.2: Eqs 8.7-8.10 (chain-rule split, delta_j z_i), 8.11 (delta_k = y_k - t_k), 8.13 (backprop formula),
  Algorithm 8.1, Eq 8.14 (sum over n); 8.1.3 A simple example; 8.1.4 Numerical differentiation (O(W) vs
  O(W^2), Fig 8.2); 8.2 intro (hand-coded / numerical / symbolic / AD, expression swell);
  8.2.1 forward mode (tangents, Figs 8.4 and 8.5); 8.2.2 reverse mode (adjoints, one sweep per output).
- Prince (PDF of Oct 23, 2023): 5.3.4 Heteroscedastic regression, Eq 5.15, Fig 5.5; Fig 3.13 Activation
  functions; 7.4.1 backprop summary (matrix form); 7.4.2 Algorithmic differentiation; 7.4.3 Extension to
  arbitrary computational graphs; Fig 7.9 Computational graph (problem 7.12); 7.5 Parameter initialization
  (vanishing/exploding), Fig 7.7; chapter-7 Notes "Reducing memory requirements" (checkpointing,
  reversible networks) on printed page 114.
- Baydin et al. (JMLR PDF fetched): 3.1.1 Dual Numbers, 3.2 Reverse Mode (cost c < 6 ops(f)), 5 Implementations.
- Baur & Strassen (ScienceDirect open-archive full text fetched): Theorem 1 (L <= 3L, nonscalar) is sec 2
  p. 318; Theorem 2 (all operations counted, L1 <= 4 L1(f)) is sec 4 "An extension", p. 329. The slide's
  "at most about 4C" counting arithmetic operations is Theorem 2 -> section 4.

## Judgement calls
- Visible captions name Prince figures by numbers that differ from the edition we cite:
  "Prince Fig. 5.14" (heteroscedastic) is **Figure 5.5** here (5.14 is the mixture-of-Gaussians figure);
  "Prince Fig. 7.8" (computational graph: exp, (.)^2, sin) is **Figure 7.9** here (7.8 is sample code).
  Crops checked visually. Tagged with this edition's numbers; visible captions left unchanged.
  "Prince Fig. 7.7" (forward/backward pass variance) matches.
- "The one idea": visible tag says "B&B 8.3-8.5"; in this edition the chain-rule split is Eqs 8.7-8.10
  (8.3-8.4 are the single-layer case, 8.5 the pre-activation). Tagged equation 8.7-8.10.
- Heteroscedastic loss on the slide uses exp(.) for sigma; Prince Eq 5.15 squares f2. Still tagged 5.15
  as the loss the slide adapts.
- Sigmoid bound slides ("What backpropagation actually uses"/"Saturation, drawn", "Why the logistic sigmoid
  fails at depth", B "Differentiated: smooth and saturating"): the 1/4 bound and the 4^-L corollary are
  the course's own derivation from Bishop Eq 5.72, motivated by 6.2.3's vanishing-gradient remark; tagged
  equation 5.72 + section 6.2.3.
- Dead-ReLU slides: Bishop 6.2.3 ("some hidden units receive no error signal", leaky ReLU fix).
- B activation zoo: the course's own plots of ReLU/leaky/ELU/softplus and GELU/SiLU/tanh/logistic. Not a
  redraw of Bishop Fig 6.12 (ELU, GELU, SiLU, logistic are not in it); Prince Fig 3.13 shows most of them.
  Tagged bishop 6.2.3 + prince Fig 3.13 (order by which book covers more of the panel's set).
- B "Measured, over twenty layers" and B "The computational graph" / "Which mode, and why": course's own
  plot / TikZ, not redraws of a book figure; tagged by section only (Prince 7.5; Prince 7.4.3, 7.4.2;
  Bishop 8.2.2, 8.2.1).
- B "When the target is genuinely multi-valued": recreated plot uses Bishop's toy problem (Fig 6.17) and
  shows what Fig 6.19 shows in panel (c); tagged figure 6.17, figure 6.19, eq 6.38, sec 6.5.
- "A worked pass, by hand": the course's own 1-2-1 network, but it is Bishop 8.1.3's simple example in
  miniature; tagged section 8.1.3.
- "The algorithm": Bishop Algorithm 8.1 (+ Eq 8.14 for the batch sum) is the main source; the vector form
  matches Prince 7.4.1, cited second.
- "The cheap gradient principle": baur1983 sec 4 (Theorem 2), then baydin2018 sec 3.2 (reverse-mode cost
  and the storage price).
- \nosrc: Outline, Notation, "Six lectures...", "Four questions", "What the activation is for" (course
  framing), "Choosing an activation" (summary table), "What Class 6 established" (recap), "Why this matters
  for training" (course discussion), "The problem, stated" (motivation), "One training step, end to end",
  Takeaways, References, Questions.

## Unverified
- none.
