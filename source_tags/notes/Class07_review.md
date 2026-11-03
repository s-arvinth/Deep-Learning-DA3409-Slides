# Class 07 review (independent reviewer)

Decks: A_Textbook_Figures (24 frames, 39 \src, 7 \nosrc), B_Recreated_Figures (42 frames, 53 \src, 12 \nosrc).

## Checked
- Prince 2023 ch. 4: Figs. 4.1, 4.2, 4.3 and Eqs. 4.1-4.4 (§4.1); Fig. 4.4 (§4.2/4.3), Eqs. 4.7-4.9, Fig. 4.5 (§4.3);
  §4.3.1 (hyperparameters: depth, width); Eq. 4.14 (§4.4, the vector form shown on A "Two hidden layers");
  Fig. 4.6, Eqs. 4.15-4.16 (§4.4.1); §4.5.1, Fig. 4.7 / §4.5.2 ((D+1)^K regions), §4.5.3, §4.5.5.
- Bishop 2024 ch. 6: Fig. 6.8 (§6.1.3), Eq. 6.19 and layer-counting text and "exponential number of hidden units"
  (§6.3 intro), §6.3.1, §6.3.2, §6.3.3 (text matches slide bullets), Fig. 6.11 (in §6.2.2, used for §6.3.3),
  Fig. 6.15 (§6.3.6). §6.2 "extends recursively" quote.
- Papers opened by the reviewer:
  - cybenko1989 (njit.edu copy, journal pages 303-314): Theorem 2 in §3; sine/cosine via Stone-Weierstrass in §4
    (pp. 310-311). Tags §3 and §4 correct.
  - leshno1993 (pinkus.net.technion.ac.il copy): Theorem 1 in §4 "Results" (p. 863); proofs in §6 (Step 1 polynomial
    case, Step 2 ridge reduction). Tags §4 and §6 correct.
  - arora2018 (arXiv v6, ICLR version): Theorem 2.1 (ReLU DNN = PWL, depth <= ceil(log2(n+1))+1) in §2 (no
    subsection before it). Tag §2 correct. (NB in arXiv v5 the same result is Corollary 2.2 in §2.2.1; v6 is the
    ICLR version and is the one to cite.)
  - telgarsky2016 (PMLR v49 PDF): Theorem 1.1 in §1.1 "Main result"; Lemmas 3.8, 3.10, 3.11 (sawtooth
    sigma(2sigma(z)-4sigma(z-1/2))) in §3.3 "Many layers, many oscillations". Tags correct.
  - montufar2014 (arXiv v1): region lower bound (Theorem 3 in v1; product of floor(n_i/n0)^n0 times sum binom) in
    §3.2 "Formal Result". Tag §3.2 correct (section number same as reported for v2).
- \srcdef lines (B): arora2018, cybenko1989, hornik1991, leshno1993, montufar2014, telgarsky2016 — text identical to
  keys.md, all complete, no unused or missing keys.

## Changes
None.

## Checks
check_sources: A 0 errors / 0 warnings, B 0 errors / 0 warnings. verify_book_numbers: 0 not found.
build + compare_pdfs: Class07A IDENTICAL (29 pages), Class07B IDENTICAL (63 pages).

## Unverified (by this reviewer)
- B "Universal approximation in $D$ dimensions": \src{hornik1991}{section}{2}. Could not open a copy of Hornik 1991
  (fetches timed out); the theorem's content (Hornik Theorem 2: continuous, bounded, nonconstant -> dense in C(X)) is
  confirmed via Leshno et al. p. 862, but the section number rests on the tagger's reading.
