# Class 05 — Shallow NN: Regression

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 34 | 73 | 6 | 0 errors, 0 warnings | IDENTICAL (39 pages) |
| B_Recreated_Figures | 42 | 77 | 11 | 0 errors, 0 warnings | IDENTICAL (54 pages) |

`\srcdef` keys: A `cybenko1989`, `hornik1991`; B `cybenko1989`, `hornik1991`, `leshno1993` (all existing
keys in keys.md). Fixed keys `bishop2024`, `prince2023`. All tags are the first lines of the frame body.

## Verification
- Prince (prince2023udl.pdf): figures 2.1, 2.2, 2.3, 2.4, 3.1-3.13 (captions read); equations 2.4, 2.5,
  3.1-3.10; sections 2.2.1, 2.2.2, 2.2.3, 3.1, 3.1.1, 3.1.2, 3.2, 3.3.1, 3.3.2, 3.4, 3.5; chapter 3 opening;
  Notes p. 37 (activation functions, Fig. 3.13) and p. 38 (Cybenko/Hornik attribution; Zaslavsky bound).
  Page numbers from the printed running heads ("Notes 37", "38 3 Shallow neural networks").
- Bishop (2024 PDF): equations 4.3, 4.8-4.16, 4.37, 4.39, 6.7-6.9, 6.12, 6.17, 6.29; figures 4.1, 4.2,
  4.3, 4.5, 6.9, 6.10, 6.12, 6.17; sections 4.1.1-4.1.4, 4.2, 6.2, 6.2.1-6.2.4, 6.4.1, 6.5.1.
- Cybenko 1989 (open copy, journal pagination): Theorem 2 in §3 "Application to Artificial Neural
  Networks" (p. 308); existence-only / curse-of-dimensionality remark in §5 (p. 313).
- Hornik 1991 (open copy): Theorem 2 (continuous, bounded, nonconstant -> dense in C(X)) in §2 "Results".
- Leshno et al. 1993 (pinkus.net.technion.ac.il, journal version): Theorem 1 (dense iff not a polynomial)
  in §4 "Results"; "if sigma is a polynomial ... not dense" is Step 1 of the proof in §6 "Proofs".

## Judgement calls (edition differences: the visible text is not changed)
- Prince figure numbers: the A deck's visible captions say Fig. 2.2 for SupervisedLinear, Fig. 3.12 for
  ShallowNetThreeInputsTwoOutputs and Fig. 3.11 for ShallowTerminology. In the Oct-2023 PDF these are
  Fig. 2.1 (family of lines), Fig. 3.11 (3 inputs, 2 outputs, 20 parameters) and Fig. 3.12
  (Terminology). Images checked; tags use the PDF numbers.
- Bishop numbering: the decks' "Eqs. 6.1-6.3", "§6.1" and "§6.1.1" are another edition's numbers. In the
  2024 PDF the two-layer network is Eqs. 6.7-6.9 in §6.2 (6.1 = "Limitations of Fixed Basis Functions"),
  weight-space symmetries are §6.2.4, and the network sum-of-squares error with its non-convexity
  remark is Eq. 6.29 in §6.4.1. Tags use the PDF numbers.
- "Why least squares?": visible "§4.1.1-4.1.3"; Gaussian noise and ML are §4.1.2-4.1.3 (§4.1.1 is basis
  functions), Fig. 4.5 sits in §4.2. Tagged eqs 4.8-4.11, Fig. 4.5, §4.1.2, §4.1.3 (+ §4.2 in A).
- "Notation: two textbooks, same model" (A) and "Notation used throughout this course" (B): `\nosrc`
  (notation slides, per guide).
- "Why shallow networks": visible "Prince Ch. 3 opening" -> `\src{prince2023}{chapter}{3}`.
- "Universal approximation" (A): Prince Fig. 3.5, §3.2, then Cybenko §3 and Hornik §2 (named on the
  slide as the theorem's source). B "Universal approximation: the statement": the theorem as stated
  (continuous, not a polynomial) is Leshno's Theorem 1 -> leshno1993 §4 first, then Cybenko §3, Hornik §2
  (in the theorem title), Leshno §6 (polynomial necessity).
- B "Lemma 1", "Lemma 2", "Proof of the theorem for D=1", "The construction, drawn", "Using the theorem:
  how fast does the error fall?": `\nosrc` — the course's own constructive proof, plot and rate.
- B "What the theorem does not say": Cybenko §5 (existence only, feasibility/number of units, curse of
  dimensionality). B "The question": Prince §3.2 (each unit adds a region).
- "How many regions?": Prince Figs 3.10/3.9 (A) or 3.9 (B plot (a) recreates 3.9a), §3.3.2 (where the
  figures are discussed), p. 38 (Prince's statement of Zaslavsky's bound). No zaslavsky1975 tag (Unverified).
- "Other activation functions": Prince Fig. 3.13 + Notes p. 37. B plot is a recreation of the same set.
- "It works" (A): Bishop Fig. 6.17 sits in §6.5.1; NB its right panel is the *inverse* problem fit badly,
  not a second good fit as the caption implies. B "It works": the course's own fits (x^2, sin), tagged
  with the figure the A crop shows (Fig. 6.17, §6.5.1), per the guide's A/B rule.
- B recreations -> book figures: linear_fit (data + residuals) -> Prince 2.2; loss_surface -> Prince 2.3
  (surface) and 2.4 (descent path); basis_functions -> Bishop 4.2; conditional_mean -> Bishop 4.5;
  LR network TikZ -> Bishop 4.1; relu -> Prince 3.1; buildup -> 3.3; family -> 3.2; depicting TikZ -> 3.4;
  learned_basis -> Bishop 6.10; two_outputs -> 3.6; two-input TikZ -> 3.7; two_inputs, buildup2d -> 3.8;
  general case TikZ -> 3.11 (+ Bishop 6.12 matrix form, §6.2.1 absorbed biases); terminology TikZ -> 3.12.
- B "Result: the normal equations": Bishop eqs 4.12-4.16 (gradient, normal equations, solution, design
  matrix, pseudo-inverse) and §4.1.3. B "What is the best possible prediction?": eqs 4.37, 4.39, §4.2.
- Outline, The dictionary, Takeaways, References, Questions?: `\nosrc`.

## Unverified
- Zaslavsky (1975), AMS Memoirs 154 ("How many regions?", A and B): paywalled, not opened; the bound is
  tagged via Prince p. 38, which quotes it; no zaslavsky1975 tag added.
