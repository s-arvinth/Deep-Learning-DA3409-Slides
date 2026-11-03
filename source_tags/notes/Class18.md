# Class 18 — Regularization: Explicit and Implicit

## Counts
- A_Textbook_Figures: 35 frames, 66 \src, 9 \nosrc. PDF IDENTICAL to baseline (50 pages).
- B_Recreated_Figures: 37 frames, 71 \src, 9 \nosrc. PDF IDENTICAL to baseline (52 pages).
- All tags are the first lines of the frame body (no frame needed the end-of-frame placement).

## \srcdef keys
- A: balduzzi2017, barrett2021, bishop2006 (new, appended to keys.md), nowlan1992, simard1992, smith2021, tibshirani1996, wolpert1996
- B: the same plus he2016

## Checker note
tools/check_sources.py reports "malformed \src (needs three arguments)" for every frame that contains the deck's
existing \srcline{...} macro: its regex `\\src(?!\{)` matches the prefix of `\srcline`. These are false positives
(17 in A, 25 in B); no other error. With that one regex excluding \srcline (patched copy run
outside the repo), both decks give 0 errors, 0 warnings. The tool itself was not changed.

## Verification
- Bishop 2024 (local PDF): sections 9.1.1-9.1.4, 9.2, 9.2.1, 9.2.2, 9.4, 9.4.1, 9.5; Eqs 9.1, 9.2-9.4, 9.5, 9.6-9.15,
  9.18-9.20, 9.35-9.38, 9.40; Figs 9.2, 9.3, 9.5, 9.6, 9.12, 9.13, 9.15 — all seen printed.
- Prince 2023 (local PDF): sections 9.1, 9.1.1, 9.1.2, 9.2, 9.2.1, 9.2.2; Eqs 9.2, 9.3-9.4, 9.8, 9.9-9.10; Figs 9.1-9.5.
- barrett2021 (arXiv 2009.11162): Theorem 3.1 in Section 3; its proof in Appendix A.1.
- smith2021 (arXiv 2101.12176): modified SGD loss with the variance term is Eq. 2 (Section 1); derived in Section 2.2.
- balduzzi2017 (arXiv 1702.08591): resnet gradient analysis in Section 3.2 "Residual networks".
- he2016 (arXiv 1512.03385): Section 3.1 "Residual Learning" (identity reachable by driving weights to zero).
- tibshirani1996 (open JSTOR copy, JRSS-B 58(1) 267-288): Section 2.3 "Geometry of Lasso" (Fig. 2).
- simard1992 (NeurIPS proceedings PDF, pp. 895-903): the tangent-prop regularizer is Section 2 "Implementation".
- bishop2006: Section 3.5.3 "Effective number of parameters" seen in the PRML table of contents (Microsoft PDF).

## Judgement calls
- "Notation for today": \nosrc (notation slide) although its \srcline names Eq. 9.1 / Eq. 9.2.
- "From measuring the error...", "Three places to put a preference", "What each mechanism prefers": \nosrc (recap / summary tables).
- "Proof, and what it means" (eigenbasis shrinkage proof): \nosrc, the course's own derivation; the result itself is tagged on the previous frame.
- "Where the extra term comes from": tagged barrett2021 section A.1 — the slide's order-alpha^2 matching is the backward-error argument of that appendix, not an own derivation.
- "What weight decay does to each direction": Bishop 2024 gives the picture (Fig. 9.3, §9.2) but not the formula lambda_i/(lambda_i+alpha); bishop2006 §3.5.3 added as the srcline names it. B "The flat directions go first" (own plot of that factor) cites bishop2006 §3.5.3 first, then Bishop 2024 §9.2.
- "Why the lasso sets coefficients to zero": Bishop (Fig. 9.6, §9.2.2) placed first although the srcline lists Tibshirani first — the statement uses Bishop's q / eta notation of Eqs. 9.18-9.20.
- "What this predicts, and what is observed": §9.2.2 (srcline) plus §9.2.1, where Prince states the step-size observation; B also gets Fig. 9.5 because its srcline cites it (in A, Fig. 9.5 is on the next frame).
- "Why residual connections smooth the surface": both variants tagged Eq. 9.40, Figs 9.12 and 9.15, §9.5, balduzzi2017 §3.2 (B's srcline names Figs 9.12, 9.15; A's frame body is the same, so it gets the same tags).
- B "Residual connections": the TikZ diagram is a redraw of Fig. 9.13 (the crop in A), so figure 9.13 is cited though B's srcline does not name it.
- B "The data does not determine the function" (own figure) cites only Bishop §9.1.1, as its srcline; B "A penalty reshapes the whole surface" is the recreation of Prince Fig. 9.1.
- Bronstein et al. (2021) and He et al. appear only in the References frame of A; no tag.

## Unverified
1. "Learning from data alone is not possible" — \src{wolpert1996}{page}{1341-1390}: the paper (Neural Computation 8(7)) is paywalled and could not be opened; the range is the article's page range (from the DOI / bibliographic records), not the page of the theorem.
2. "Parameter sharing: a hard constraint" — \src{nowlan1992}{page}{473-493}: paywalled, could not be opened (author copy at cs.toronto.edu returns unparsed binary); range is the article's page range from bibliographic records.
