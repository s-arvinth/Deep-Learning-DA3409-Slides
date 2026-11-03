# Class 19 — Regularization: Heuristic Methods

## Counts
- A_Textbook_Figures: 27 frames, 46 \src, 7 \nosrc. PDF IDENTICAL to baseline (43 pages).
- B_Recreated_Figures: 30 frames, 61 \src, 7 \nosrc. PDF IDENTICAL to baseline (46 pages).
- All tags are the first lines of the frame body.

## \srcdef keys (both variants)
bishop1995a, bishop1995b, breiman1996, srivastava2014 (all existing keys in keys.md, same text).

## Checker note
tools/check_sources.py reports "malformed \src" for every frame containing the deck's own \srcline{...} macro
(regex `\\src(?!\{)` matches the prefix of `\srcline`): 11 such errors in A, 14 in B, and no other errors. A copy of
the checker with only that regex excluding \srcline gives 0 errors, 0 warnings for both. The tool was not changed.

## Verification
- Bishop 2024 (local PDF): §9.1.3, §9.3.1, §9.6, §9.6.1; Eqs 9.42-9.43, 9.44-9.50, 9.51; Figs 9.1, 9.7, 9.8, 9.17.
- Prince 2023 (local PDF): §9.3.1-9.3.8; Eqs 9.11-9.12; Figs 9.6-9.13.
- bishop1995a: full text of the Aston technical report NCRG/95/022 (publications.aston.ac.uk/id/document/38):
  §2.3 "Early Stopping", Eqs (10)-(14), Eq (12) = w_j^(tau) = {1-(1-eta lambda_j)^tau} w*_j.
- bishop1995b: author copy (pages.cs.wisc.edu/~yliang/cs760_fall18/Bishop_Noise.pdf): §2 "Training with Noise", Eqs (7)-(19).
- breiman1996: Technical Report 421 version (cs.nyu.edu copy): bagging defined in §1 Introduction.
- srivastava2014 (JMLR PDF): 2^n thinned weight-sharing networks and test-time weight scaling in §1 (printed p. 1931).

## Judgement calls
- "Notation for today", "From the objective to the procedure", "The whole toolbox", Takeaways, References, Questions, Outline: \nosrc.
- "Stopping early is weight decay in disguise": bishop1995a first (Eq. 12 is the slide's formula; Bishop 2024 §9.3.1 states the result only qualitatively and points to Exercise 9.6), then Bishop 2024 §9.3.1 and Fig. 9.8 as in the srcline.
- "Where that comes from": tagged bishop1995a Eqs 11-14 / §2.3 — steps 1-2 are exactly Eqs 11-12 of that paper; the exponential approximation in steps 3-5 is the course's own.
- "What averaging is worth": Exercise 9.15 (the E_COM <= E_AV bound) is named on the slide but "exercise" is not an allowed type; tagged Eqs 9.44-9.50 and §9.6, where the book states the bound and points to the exercise.
- "Proof of both halves": first half is Bishop's own derivation Eqs 9.47-9.50, so tagged; the Jensen half is the exercise solution (no extra tag).
- "An ensemble you do not have to train": Bishop placed first (the figure and the "M non-output nodes, 2^M networks" wording are §9.6.1's) although the srcline names Srivastava first; B adds Prince Fig. 9.8 as its srcline does. The B TikZ drawing is a redraw of Bishop Fig. 9.17.
- B "Dropout, measured" (own experiment on dropout rate): tagged Bishop §9.6.1, whose "rho = 0.5 for hidden nodes" recommendation is the claim the caption tests.
- B "The promise, and what arrives" (own committee-error experiment): tagged the result it tests, Bishop Eq. 9.50 / §9.6, plus Prince §9.3.2 (bagging).
- B "The two filters, measured" (own experiment comparing the two filters): same tags as the theorem it tests. B "Descent on a quadratic, drawn" is a redraw of Bishop Fig. 9.8.
- B "The equivalence, checked": Prince Fig. 9.10 recreation plus a ridge check of the noise theorem, so bishop1995b §2 is added.
- B "When the data set is the constraint": srcline gives "§9.3.6-9.3.8"; written as three \src lines (9.3.6, 9.3.7, 9.3.8) rather than a section range.
- Szegedy et al. (label smoothing), Goodfellow et al. (adversarial), Gal & Ghahramani, Freund & Schapire appear only in References; no tags.

## Unverified
1. bishop1995a (§2.3, Eqs 11-14 / 12): checked in the technical report NCRG/95/022, not in the printed ICANN'95 proceedings (pp. 141-148); section/equation numbering of the printed version not seen.
2. bishop1995b (§2): checked in the author's postprint, not the Neural Computation typeset version (section structure assumed the same).
3. breiman1996 (§1): checked in UC Berkeley Technical Report 421 (1994), not the Machine Learning 24(2) article.
