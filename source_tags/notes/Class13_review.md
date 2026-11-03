# Class 13 review (Gradient Descent and Optimization)

A: 45 frames (before review 65 \src + 11 \nosrc); B: 51 frames (70 \src + 12 \nosrc). All tag lines and both \srcdef lines checked.
(The checker now reports 0 errors for both decks; the \srcline false positive in the tagger's notes is gone.)

## How checked
- Bishop (booktext): ch. 7 intro ("goal is never simply exact optimization"); 7.1 with Eqs 7.1-7.2, Fig 7.1 and the M! 2^M
  remark; 7.1.1 Eqs 7.3-7.14 and Fig 7.2; 7.2 Eq 7.15; 7.2.1 W(W+3)/2, O(W^3) vs O(W^2); 7.2.2 Eq 7.16; 7.2.3 Eqs 7.17-7.18,
  Algorithm 7.1, duplication argument, escaping local minima; 7.2.4 sigma/sqrt(N), 100 -> 10, powers of 2, shuffling,
  Algorithm 7.2; 6.2.4 weight-space symmetries. All correct.
- Prince: 6.1 Eqs 6.1-6.3 and the line-search remark; 6.1.1 Fig 6.1; 6.1.2 chord definition, Figs 6.2, 6.3; 6.1.3 Fig 6.4;
  6.2 / Fig 6.5; 6.2.1 Eq 6.10, Fig 6.6, "expected gradients ... remain the same"; 6.2.2 the six properties and "does not
  necessarily converge"; printed p. 92 Notes (convexity via the Hessian, classification of stationary points, line search /
  bracketing Fig 6.10); Fig 6.11 (problem 6.6). All correct.
- Bottou 2010: 2.2 Stochastic gradient descent, Eq (4) w_{t+1} = w_t - gamma_t grad Q(z_t, w_t); pages 177-186 from Crossref.

## Changes (A and B identically)
1. "Stochastic gradient descent": added \src{bottou2010}{equation}{4} before \src{bottou2010}{section}{2.2}
   (less specific: the SGD update is Bottou's Eq. 4).
2. srcdef bottou2010 (A, B, keys.md): "COMPSTAT 2010, 2010" -> "Proceedings of COMPSTAT 2010, 177-186, 2010" (pages missing).
3. srcdef nesterov2004 (A, B, keys.md): "Kluwer / Springer, 2004" -> "1st edition, Kluwer / Springer (Applied Optimization 87),
   2004" (book edition missing; now identical to the text Class 14 already uses for the same key).

Counts after review: A 66 \src + 11 \nosrc; B 71 \src + 12 \nosrc.

## Checks
check_sources: A 0/0, B 0/0; verify_book_numbers: 0 not found; compare_pdfs: A IDENTICAL (61), B IDENTICAL (66).

## Still unverified
Nesterov (2004) could not be opened (Springer chapter PDFs are paywalled; only the chapter list and the Scribd 1998-draft TOC are
visible). The nine \src{nesterov2004}{section}{1.2.2} tags stay as the tagger listed them:
A "Four regimes for the step size", A "Smoothness: one number for the worst curvature", A "How far a single step may go",
B "Smoothness: ...", B "How far a single step may go", B "The guarantee, as a function of the step", B "Four regimes for the
step size", B "The four regimes on the contour map", B "The four regimes on the surface".
Additional concern for whoever checks a copy: in the 1998 draft, 1.2.2 "Classes of differentiable functions" holds the
Lipschitz-gradient class and the quadratic upper bound (Lemma 1.2.3), but the one-step decrease f(x) - h(1 - hL/2)||f'(x)||^2,
the optimal step h = 1/L and the step-size discussion sit in 1.2.3 "Gradient method". If the 2004 edition is the same, the
one-step-lemma and step-regime frames should also (or instead) cite section 1.2.3; the smoothness frames keep 1.2.2.
