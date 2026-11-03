# Class 07 — Deep Networks, Universal Approximation

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 24 | 39 | 7 | 0 errors, 0 warnings | IDENTICAL (29 pages) |
| B_Recreated_Figures | 42 | 53 | 12 | 0 errors, 0 warnings | IDENTICAL (63 pages) |

`\srcdef` keys: A none (fixed keys only). B: arora2018, cybenko1989, hornik1991, leshno1993, montufar2014,
telgarsky2016 — all existing keys from keys.md, same reference text. No new keys created.
All tags are the first lines of the frame body.

## Verification
- Prince 2023 PDF: figures 4.1-4.7; equations 4.1-4.4, 4.7-4.9, 4.14, 4.15, 4.16; sections 4.1, 4.3,
  4.3.1, 4.4.1, 4.5.1, 4.5.2, 4.5.3, 4.5.5.
- Bishop 2024 PDF: figures 6.8 (in §6.1.3), 6.11 (in §6.2.2), 6.15 (in §6.3.6); equation 6.19 (§6.3);
  sections 6.1.3, 6.2, 6.3, 6.3.1, 6.3.2, 6.3.3, 6.3.6.
- Papers, opened and read:
  - cybenko1989 (copy at papers.baulab.info, journal pagination 303-314): Theorem 2 (the sigmoidal UA
    theorem) is in §3 "Application to Artificial Neural Networks", p. 308; sine/cosine density via
    Stone-Weierstrass is in §4 "Results for Other Activation Functions", pp. 310-311.
  - hornik1991 (copy at papers.baulab.info, pp. 251-257): Theorem 2 (continuous, bounded, nonconstant ->
    dense in C(X)) in §2 "Results", p. 252.
  - leshno1993 (pinkus.net.technion.ac.il, journal pagination 861-867): Theorem 1 in §4 "Results", p. 863;
    proof (Step 1: polynomial case; Step 2: reduction to ridge functions/1-D) in §6 "Proofs", p. 864.
  - arora2018 (arXiv 1611.01491v6): Theorem 2.1 (ReLU DNN = piecewise linear, depth <= ceil(log2(n+1))+1)
    in §2.
  - telgarsky2016 (arXiv 1602.04485v2): Theorem 1.1 in §1.1 "Main result"; sawtooth
    f(z)=sigma(2sigma(z)-4sigma(z-1/2)) and its k-fold composition: Lemmas 3.8, 3.10, 3.11 in §3.3
    "Many layers, many oscillations".
  - montufar2014 (arXiv 1402.1869v2): Theorem 4 (region lower bound) in §3.2 "Formal Result".
  Papers with numbered sections are cited by the theorem's section, per guide.

## Judgement calls
- "Where the shallow network left us", "An intuition: solve a big problem in stages", B "Notation",
  B "What we already proved" (Class 5 lemma): `\nosrc` (recap / own analogy / notation).
- "Shallow vs. deep: the trade-off": `\nosrc` (summary table).
- A "Two hidden layers": Prince Eqs. 4.7-4.9 (scalar form) and 4.14 (the vector form actually shown).
- A/B "The same object in Bishop's notation" / "A note on counting layers": the slide credits §6.1/§6.3;
  in the 2024 PDF the deep-network equation (6.19), the "more than two layers" text and the layer-counting
  discussion are in §6.3, and Fig. 6.15 sits in §6.3.6 (feed-forward, no cycles). Tagged accordingly.
- "Building depth recursively": the quoted "extends recursively ... hierarchical model with many layers"
  is in the §6.2 intro; the recursion itself is Eq. 6.19 in §6.3. Both tagged, 6.19 first.
- A "Representation learning": Fig. 6.11 lives in §6.2.2 but is used to illustrate §6.3.3; both tagged.
- A "Universal approximation, revisited": Prince §4.5.3 (depth efficiency), §4.5.1, Bishop §6.3
  (exponential regions/units). B "What universality does not settle" is the instructor's own list -> `\nosrc`.
- "Why depth tends to win in practice": Prince §4.5.5 (training/generalisation) + Bishop §6.3.1
  (compositional bias).
- B figures `compose`, `fold`, `compose2d`, `buildup`, `regions` are recreations of Prince Figs 4.1, 4.3,
  4.2, 4.5, 4.7 (A-variant crops); B tikz network diagrams correspond to Prince Figs 4.1, 4.4, 4.6. Cited.
  B "Two networks to work with" uses its own numbers but Prince's set-up (Eqs. 4.1-4.4, §4.1).
- B "Why the pieces multiply" (own proposition/proof) and "Counting for our two networks": tagged with the
  Prince place they formalise (§4.1 composition argument; §4.5.2 (D+1)^K count).
- B "The general deep network": Prince Fig. 4.6 / Eq. 4.15 / §4.4.1 / §4.3.1, plus Bishop Eq. 6.19 for the
  z^(l) = h(W z^(l-1) + b) notation used on the slide.
- B "Ridge functions" (definition): `\nosrc` (own framing; Leshno Remark 2 mentions ridge functions only
  in passing).
- B "Ridge functions are enough" (trig ridge density by Stone-Weierstrass) and "The construction, drawn"
  (own experiment illustrating that step): cybenko1989 §4, where Cybenko states exactly this
  (sin/cos completeness by Stone-Weierstrass). The proof on the slide is the instructor's.
- B "Universal approximation in D dimensions": theorem titled "Cybenko 1989; Hornik 1991" -> cybenko1989 §3
  (Thm 2), hornik1991 §2 (Thm 2); the stated hypothesis (non-polynomial) and the ridge-reduction proof
  follow Leshno §6 (Step 2) -> leshno1993 §6 added third.
- B "Which activations work?": leshno1993 §4 (Theorem 1) and §6 (Step 1 is the "only if" proof shown).
- B sawtooth frames: the lemma slide does not name Telgarsky, but T(x)=2h(x)-4h(x-1/2) is Telgarsky's
  Lemma 3.10/3.11 -> telgarsky2016 §3.3; "The sawtooth, drawn" (own plot) tests that lemma, same tag.
  "Depth beats width" is "after Telgarsky": §1.1 (Thm 1.1) and §3.3. The exact-representation argument on
  the slide is a simplification by the instructor.
- B "A hierarchy, built explicitly": `\nosrc` (own constructed example).
- B "Representation learning: the set-up/result" and "What that experiment shows": own experiment testing
  Bishop §6.3.3 -> `\src{bishop2024}{section}{6.3.3}`; Fig. 6.11 not cited in B (the B plot is not a
  recreation of it).
- Visible citations in References frames (Cybenko, Hornik, Montufar in A) are not tagged (References = `\nosrc`).

## Unverified
(none — all six papers were opened; the Cybenko/Hornik copies are third-party hosted scans carrying the
journal pagination, not the publisher's site.)
