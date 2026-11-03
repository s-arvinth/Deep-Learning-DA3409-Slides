# Class 26 — BPTT and Bidirectional RNNs

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 36 | 73 | 7 |
| B_Recreated_Figures | 41 | 89 | 6 |

Checks: `tools/check_sources.py` reports ERRORs "malformed \src (needs three arguments)" (A 21, B 22), all on frames containing
the deck's `\srcline{...}` macro — checker false positive (`\\src(?!\{)` matches `\srcline`). With the regex also excluding
`\srcline` (patched copy in /tmp; tool not modified): 0 errors both. A has 1 WARN: References frame body starts with
`{\scriptsize`, so its `\nosrc` is the last line before `\end{frame}` (the guide's known case). PDFs: both IDENTICAL to baseline
(A 53 pages, B 58 pages).

## \srcdef keys
A and B: elman1990, jurafsky_slp3 (guide's August 19, 2026 draft text), schuster1997, werbos1990 (all existing keys; no new key)

## Where the numbers were checked
- Jurafsky & Martin aug26 draft: Ch. 14 "RNNs and LSTMs" opening ("inherently temporal"); 14.1, 14.1.1, 14.1.2 (fixed-length segments = truncation
  "in words"), 14.2, 14.3, 14.4.2, 14.5 (example (14.19) "The flights the airline was canceling were full"), 14.6; Eqs (14.1)–(14.3) in 14.1.1,
  (14.16)–(14.18) in 14.4.2; Figs 14.1–14.5, 14.11, 14.15.
- Goodfellow: Ch. 10 opening; 10.1, 10.2 (O(τ) runtime and memory), 10.2.2, 10.3, 10.7; Eqs (10.1)–(10.7) in 10.1, (10.8)–(10.14) in 10.2,
  (10.17)–(10.28) in 10.2.2; Figs 10.3 (10.2), 10.4, 10.5, 10.11.
- elman1990: scanned journal copy (papers.baulab.info) — "Networks with memory" and the simple recurrent network with context units,
  printed pp. 182–184 (Fig. 2 on p. 184).
- werbos1990: Zenodo copy of the Proc. IEEE article — section "III. Backpropagation through time".
- schuster1997: journal copy — §II.B "Bidirectional Recurrent Neural Networks" (structure, training; Fig. 3), pp. 2675–2676.

## Judgement calls
- \nosrc: Outline, Notation, "The block ahead, and where today sits" (course-plan table; its \srcline is "Course plan … §14.6"),
  "The block ahead, revisited", Takeaways, References (A), Questions.
- "Where we are coming from": half recap of Classes 23–25, but the bullets are the chapter openings → goodfellow2016 chapter 10, jurafsky_slp3 chapter 14.
- "Ch. 14 opening" / "§10 opening" tagged as `chapter` 14 / 10 (no section number exists for the opening text).
- Own schematics in both variants (space_vs_time, running_example) tagged with the sections their captions cite.
- Example (14.19) is numbered in the equation sequence but is a sentence, not an equation: tagged as section 14.5 instead of `equation`.
- Captions "after Elman (1990)" on J Fig 14.1: attribution of the book figure, not tagged; elman1990 is tagged only on "A network with a cycle", whose \srcline names it.
- A "Unfolding, drawn" shows only G10_3 (caption says Figs 10.2–10.3): tagged figure 10.3 only; B's recreation is a "Layout of Figs 10.2–10.3": both figures.
- B measured frames ("The theorem, checked", "The terms of the sum", "The cost, measured", "Both directions, measured"): the course's own experiments;
  tagged with the result they test (Eqs 10.17–10.28 / 10.20, 10.26, §10.7 / §10.2 / §10.3).
- B "Why the tools we have fall short" uses an own figure (window_fails), not Fig 14.5, so sections 14.2 and 14.5 only (A, with the crop, cites figure 14.5).

## Unverified
- "Truncation" (A and B): \srcline names Williams & Peng (1990), Neural Computation 2, 490–501. No open copy could be opened (Williams's
  publications page lists it but gives no reachable link); no section/page verified, so no williams1990 tag was added — frame tagged jurafsky_slp3 section 14.1.2 only.
