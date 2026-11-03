# Class 09 review (independent reviewer)

Decks: A_Textbook_Figures (23 frames, 29 \src, 10 \nosrc), B_Recreated_Figures (22 frames, 27 \src, 10 \nosrc).
B = A minus "Universal approximation, illustrated"; all shared frames carry identical tags (checked by script).

## Checked
- Bishop 2024: Fig. 6.9 (§6.2) and Eq. 6.19 (§6.3); §6.1.1 "The curse of dimensionality" (grid of basis-function
  cells, exponential in D) and Fig. 6.3 in it; Eq. 6.5 (1-(1-eps)^D), Figs. 6.4, 6.5 in §6.1.2; Fig. 6.10 (x^2,
  sin, |x|, Heaviside; three tanh units) in §6.2.2.
- Prince 2023: Fig. 4.6, §4.4.1.
- Papers opened by the reviewer:
  - leshno1993: Theorem 1 ("dense iff not a polynomial") in §4 — matches "The classical statement".
  - cybenko1989: Theorem 2 in §3. hornik1991: see Unverified.
  - lu2017 (arXiv v3): Theorem 1 (width n+4, L1(R^n)) in §3 "Width-bounded ReLU Networks as Universal Approximator".
  - hanin2017 (arXiv v2): Theorem 1 stated in §1; lower-bound proof (width d_in fails, bounded level set,
    f = sum (x_j-1/2)^2) is §3 "Proof of the Lower Bound in Theorem 1". Tag §3 correct.
  - johnson2019 (arXiv v1): Theorem 1 (width <= n cannot approximate functions with a bounded level-set component)
    stated in §4 "Outline of the Main Result". Tag §4 correct.
- \srcdef lines (A and B): barron1993, cybenko1989, devore1989, hanin2017, hornik1991, johnson2019, leshno1993, lu2017 —
  identical to keys.md, complete, none missing or unused.
- \nosrc frames (Outline, Notation, recap, road-map table, definition, summary table, What none of them say,
  Takeaways, References, Questions): allowed by the guide.

## Changes
None.

## Checks
check_sources: A 0 errors / 0 warnings, B 0 errors / 0 warnings. verify_book_numbers: 0 not found.
build + compare_pdfs: Class09A IDENTICAL (31 pages), Class09B IDENTICAL (30 pages).

## Unverified (by this reviewer)
- \src{barron1993}{section}{III} / {section}{IV} (frames "Two objects...", "A rate that does not see the dimension",
  "The rate, measured", "The two bounds side by side"): no copy of Barron 1993 could be opened (fetch timed out);
  section numbers rest on the tagger's reading of a scan.
- \src{devore1989}{section}{4} / {section}{1} ("Two objects...", "And a lower bound that does", "The two bounds side
  by side"): no copy opened; rests on the tagger's reading.
- \src{hornik1991}{section}{2} ("The classical statement"): as in Class 07, not re-opened.
- Note (not a tag issue): the reference spreadsheet lists Hanin & Sellke and Johnson under "Width as a threshold: the
  upper bound"; in the deck their results (width <= D fails) are on the "lower bound" frame, which is where they are
  tagged, correctly.
