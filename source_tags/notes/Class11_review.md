# Class 11 review (Activations, Error Functions, Backpropagation, AD)

A: 39 frames (before review 50 \src + 13 \nosrc); B: 40 frames (48 \src + 13 \nosrc). All tag lines and both \srcdef lines checked.

## How checked
- Bishop (booktext): Fig 6.12 panels (a tanh, b hard tanh, c softplus, d ReLU, e leaky ReLU, f absolute) in 6.2.3; Eq 5.72;
  5.4.6 with Eq 5.95; 6.5 with Figs 6.17, 6.18, 6.19 and Eq 6.38; 8.1.2 Eqs 8.5-8.14 and Algorithm 8.1; 8.1.3; 8.1.4 with
  Fig 8.2 and O(W) vs O(W^2); 8.2 intro (expression swell); 8.2.1 (tangents, Eqs 8.57, 8.66, Figs 8.4, 8.5); 8.2.2 (adjoints,
  Eqs 8.68-8.69); 7.4.2 Eq 7.51 (vanishing/exploding gradients as a product of layer Jacobians).
- Prince: 5.3.4, Eq 5.15, Fig 5.5; Fig 3.13; 7.4.1-7.4.3; 7.5; Fig 7.7; Fig 7.9; chapter-7 Notes "Reducing memory
  requirements" on printed p. 114.
- Baydin et al. (JMLR PDF): 3.1.1 Dual Numbers, 3.2 Reverse Mode, 5 Implementations. Baur-Strassen: "4. An extension",
  Theorem 2 (all operations counted) -> section 4. All correct.

## Changes (A and B identically)
1. "Reverse mode": added \src{bishop2024}{equation}{8.68-8.69} before \src{bishop2024}{section}{8.2.2}
   (less specific: the displayed adjoint recursion is Bishop Eq 8.69, with the adjoint definition 8.68).
2. "The recursion has a price: vanishing and exploding": old {prince2023 section 7.5} -> new
   {bishop2024 equation 7.51; bishop2024 section 7.4.2; prince2023 section 7.5} (missing source: the proposition, a product of
   per-layer Jacobian factors that vanishes below 1 and explodes above 1, is Bishop 7.4.2 / Eq 7.51 nearly verbatim; Prince 7.5
   kept as second source).

Counts after review: A 53 \src + 13 \nosrc; B 51 \src + 13 \nosrc.

## Notes for the coordinator (visible text, not changed)
- "Which mode, and why" caption calls Bishop Fig 8.5 "reverse mode"; in the book Fig 8.4 and Fig 8.5 are both forward-mode
  evaluation traces (8.5 = the two-output extension). The tags (figure 8.4, 8.5) are right for the crops; the caption text is not.
- Visible "Prince Fig. 5.14" / "Fig. 7.8" captions differ from this edition's 5.5 / 7.9 (already noted by the tagger).

## Checks
check_sources: A 0/0, B 0/0; verify_book_numbers: 0 not found; compare_pdfs: A IDENTICAL (54), B IDENTICAL (55).

## Still unverified
- none.
