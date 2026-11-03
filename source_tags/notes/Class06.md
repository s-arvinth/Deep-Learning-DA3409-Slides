# Class 06 — Shallow NN: Classification

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 26 | 45 | 9 | 0 errors, 0 warnings | IDENTICAL (33 pages) |
| B_Recreated_Figures | 33 | 56 | 10 | 0 errors, 0 warnings | IDENTICAL (50 pages) |

`\srcdef` keys used: none (only fixed keys `bishop2024`, `prince2023`). No new keys added to keys.md.
All tags placed as the first lines of the frame body (no frame body starts with `{`).

## Verification
Every number was looked up in the guide's PDFs (pdftotext): Bishop & Bishop 2024 (figures 5.1, 5.4,
5.13, 5.14, 5.15; equations 5.2, 5.4, 5.5, 5.40-5.44, 5.48, 5.49, 5.72, 5.74-5.78, 5.80, 5.81; sections
5.1.1, 5.1.4, 5.3.1, 5.4.2, 5.4.3, 5.4.4) and Prince 2023 (figures 5.4, 5.6-5.12; equations 5.6, 5.11,
5.17, 5.21, 5.27-5.31; sections 5.1, 5.2, 5.3.1, 5.4, 5.5, 5.5.1, 5.7). The section of each figure /
equation is the section whose text it sits in.

## Judgement calls
- Recap: the shallow network (A and B): `\nosrc` (recap, per guide), although the equations are Bishop's.
  NB the visible text in A says "Eqs. 6.1--6.3"; in the 2024 PDF these equations are (6.7)-(6.9) (§6.2).
  Not changed (only lines may be added).
- What changes / Output activations: the complete table / The dictionary: `\nosrc` (summary tables),
  even though "complete table" carries a visible `\figsource` pointing to B&B §5.4, §6.1, Prince §5.3-5.5.
- Why not squared error on probabilities? (A and B): `\nosrc` — the SSE-vs-CE gradient comparison is the
  course's own derivation (and B's `saturation` plot is the course's own); no book place is named.
- "You already know one instance" (A): the ML strip mentions "B&B §4.1.1" — treated as a passing mention,
  not tagged (NB in the 2024 PDF Gaussian noise -> squared error is §4.1.2 "Likelihood function"; §4.1.1 is
  "Basis functions"). The frame is tagged with Prince Fig. 5.4 / Eq. 5.11 / §5.3.1.
- Step 2: the logistic sigmoid (A): figure is Prince 5.7 (main, Prince's teaching sequence), formulas are
  Bishop's (5.42, symmetry 5.43, derivative 5.72) — all tagged, Prince first.
- B variant figures are the course's recreations of the A-variant crops, frame for frame (discriminant ->
  B&B 5.1, lsq_vs_logistic -> B&B 5.4, gaussian_posterior -> B&B 5.13, gaussian_ml -> Prince 5.4,
  bernoulli -> Prince 5.6, sigmoid -> Prince 5.7, binary_model -> Prince 5.8, categorical -> Prince 5.9,
  softmax_posteriors -> B&B 5.14, multiclass_model -> Prince 5.10, output_distributions -> Prince 5.11,
  cross_entropy_kl -> Prince 5.12, feature_space -> B&B 5.15); each cites that book figure, per guide.
  Some recreations add panels (entropy map, learned tanh features, sigmoid derivative).
- B "Step 2: the logistic sigmoid": Bishop 5.42 and logit 5.44 first (the slide's formulas), then Prince
  Fig. 5.7 for the recreated plot.
- B "How far is a point from the boundary?": Bishop Eqs. 5.4-5.5 (proof follows them), Fig. 5.1, §5.1.1
  (a commented-out `\tag{B\&B 5.7}` in the source is wrong: 5.7 is the K-class discriminant).
- B "Where the sigmoid comes from": Eqs. 5.48-5.49 (statement, labels C1/C2 swapped on the slide),
  5.40-5.41 (Bayes/log-odds step), §5.3.1.
- B "Cross-entropy: the second view": Prince Eqs. 5.27-5.31 and §5.7 (slide uses H(q,p) notation; the
  argument is Prince's).
- A "Fixed features vs learned features": "cf. Fig. 5.16" is a passing mention, not tagged.

## Unverified
(none)
