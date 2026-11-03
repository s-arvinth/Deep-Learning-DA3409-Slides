# Class 09 — MLP/FNN to Universal Function Approximation

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 23 | 29 | 10 | 0 errors, 0 warnings | IDENTICAL (31 pages) |
| B_Recreated_Figures | 22 | 27 | 10 | 0 errors, 0 warnings | IDENTICAL (30 pages) |

B is A without the frame "Universal approximation, illustrated" (Bishop Fig. 6.10); all shared frames carry
identical tags. `\srcdef` keys (both decks): barron1993, cybenko1989, devore1989, hanin2017, hornik1991,
johnson2019, leshno1993, lu2017 — all existing keys from keys.md, reference text as there. No new keys.
All tags are the first lines of the frame body.

## Verification
- Bishop 2024 PDF: Fig. 6.3 and the S^D grid argument in §6.1.1; Figs. 6.4, 6.5 and Eq. 6.5 (shell volume
  1-(1-eps)^D) in §6.1.2; Fig. 6.9 (§6.2); Eq. 6.19 (§6.3); Fig. 6.10 in §6.2.2.
- Prince 2023 PDF: Fig. 4.6, §4.4.1.
- Papers, opened and read:
  - cybenko1989 §3 (Theorem 2, p. 308); hornik1991 §2 (Theorem 2, p. 252); leshno1993 §4 (Theorem 1,
    "dense iff not a polynomial", p. 863) — see Class07 notes for the copies used.
  - barron1993 (EPFL-hosted scan, journal pagination 930-945): C_f and Theorem 1 in §III "Context and
    statement of the theorem" (p. 933); proof via Maurey's lemma in §IV "Analysis" (pp. 934-935).
    Sections are numbered with Roman numerals in the paper; tagged as printed ("III", "IV").
  - devore1989 (gwern.net scan, journal pagination 469-478): continuous selection / nonlinear n-width
    defined in §1 (eqs. 1.3-1.4, p. 470); lower bound for Sobolev balls, Theorem 4.2 (d_n >= C n^{-r/d}),
    in §4 "Lower bounds for widths of smoothness classes" (p. 475).
  - lu2017 (arXiv 1709.02540v3): Theorem 1 (width n+4, L1 universality) in §3.
  - hanin2017 (arXiv 1710.11278v2): Theorem 1 (d_in+1 <= w_min) stated in §1; the lower-bound proof
    (width d_in fails; level sets / bounded component argument) is §3.
  - johnson2019 (arXiv 1810.00393v1): Theorem 1 (width <= n cannot approximate functions with a bounded
    level-set component) stated in §4.

## Judgement calls
- Notation, "Where Class 7 left us" (recap), "Three questions, and where they are answered" (outline-like
  table; names the papers only as a road map), "What universal means" (definition), "What each theorem
  actually gives" (summary table), "What none of them say", Takeaways, References, Questions: `\nosrc`.
- "The multilayer perceptron": Bishop Fig. 6.9 + Eq. 6.19 (notation follows B&B) first, then Prince Fig. 4.6
  / §4.4.1. B's tikz diagram replaces the two A crops and gets the same tags. Parameter count W is own.
- "Why not just use fixed basis functions?": visible text cites "§6.1"; the grid/S^D argument is §6.1.1,
  tagged at that level. The proposition/proof wording is the instructor's.
- B "The curse of dimensionality, drawn" / "Two more faces": `curse_grid`, `curse_volume` recreate Bishop
  Figs 6.3 and 6.4/6.5 (panel (c) of curse_volume is own); tagged with the book figures as in A.
- "The classical statement": the theorem as stated ("dense iff h is not a polynomial") is Leshno et al.'s
  Theorem 1, so leshno1993 §4 is first; Cybenko §3 and Hornik §2 follow (title order kept after that).
- "Two objects the next two theorems need": Barron norm from barron1993 §III; Sobolev ball from devore1989
  §4 (DeVore use W^r_p balls measured in L_q; the slide's W^{s,inf} form is the instructor's specialisation).
- "A rate that does not see the dimension": barron1993 §III (Theorem 1) and §IV (proof sketch). The slide's
  constant (2 C_f r)^2 / M matches Theorem 1 on a ball B_r with |w|_B = r|w|.
- "The rate, measured" and "The two bounds side by side": own plots/experiments testing Barron (and DeVore);
  tagged with the results they test.
- "And a lower bound that does": devore1989 §4 (Theorem 4.2) then §1 (continuous selection).
- "Width as a threshold: the lower bound": title credits Hanin & Sellke first -> hanin2017 §3 (lower-bound
  proof) then johnson2019 §4 (Theorem 1; level-set argument as on the slide).
- NB: the References frame says DeVore et al. appeared in "Numer. Math. 63:469"; the paper is in Manuscripta
  Mathematica 63, 469-478 (as in keys.md). Not changed (only lines may be added).

## Unverified
(none — all eight papers were opened; Cybenko, Hornik, Barron and DeVore copies are third-party hosted scans
showing the journal pagination, not the publisher's site.)
