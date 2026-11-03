# Class 19 review (independent check of the source tags)

## Checked
- A: 27 frames, 46 \src + 7 \nosrc (before review), 4 \srcdef. B: 30 frames, 61 \src + 7 \nosrc, 4 \srcdef.
- All Bishop 2024 (§9.1.3, 9.3.1, 9.6, 9.6.1; Eqs 9.42-9.51; Figs 9.1, 9.7, 9.8, 9.17) and Prince 2023
  (§9.3.1-9.3.8, Eqs 9.11-9.12, Figs 9.6-9.13) tags read against the book text; all correct.
- \srcdef texts identical to keys.md. After edits: check_sources 0 errors / 0 warnings, verify_book_numbers 0 not
  found, PDFs IDENTICAL (A 43 pages, B 46 pages).

## Changes
1. "Stopping early is weight decay in disguise" (A, B): added \src{bishop2024}{equation}{9.56-9.61} after the
   bishop1995a tags (missing: Bishop's Exercise 9.6 prints the slide's result -- quadratic error 9.56, descent from
   the origin 9.57, w_j^(tau) = {1-(1-rho eta_j)^tau} w*_j 9.58, crossover at (rho tau)^-1 9.60-9.61; the tagger's
   note said Bishop 2024 gives it only qualitatively).
2. "Where that comes from" (A, B): added \src{bishop2024}{equation}{9.56-9.61} (missing; same derivation).
3. B "The two filters, measured" (own experiment testing that result): added \src{bishop2024}{equation}{9.56-9.61}
   (missing; keeps it equal to the theorem it tests, as the tagger intended).
4. "What averaging is worth" (A, B): added \src{bishop2024}{equation}{9.64} (missing: E_COM <= E_AV, the second
   half of the theorem, is printed as Eq. 9.64 in Exercise 9.15; the tagger had no tag for it).
5. "Proof of both halves" (A, B): added \src{bishop2024}{equation}{9.64} (missing; the Jensen half proves 9.64).
6. "Noise on the inputs is a penalty" (A, B): added \src{bishop2024}{equation}{9.52-9.53} (missing: Exercise 9.2,
   linear model 9.52 with sum-of-squares 9.53, Gaussian input noise with E[eps]=0, E[eps eps]=sigma^2 -> weight
   decay; this is the slide's theorem and its one-line proof).
Totals: missing 11 (A 5, B 6), wrong place 0, less specific 0, over-tagged 0, srcdef 0.

## Still unverified (from the tagger; could not be re-checked)
1. bishop1995a §2.3 / Eqs 11-14, 12: numbering seen in tech report NCRG/95/022 only, not the printed ICANN'95 paper.
2. bishop1995b §2: author postprint, not the typeset Neural Computation article.
3. breiman1996 §1: TR 421 version, not the Machine Learning article.
