# Class 03 — Perceptrons and the PLA

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 23 | 33 | 5 | 0 errors, 0 warnings | IDENTICAL (29 pages) |
| B_Recreated_Figures | 24 | 35 | 5 | 0 errors, 0 warnings | IDENTICAL (28 pages) |

`\srcdef` keys: none (only fixed key `rojas1996`). No new keys. All tags are the first lines of the
frame body. B has a commented-out frame "How long can it take?" (all lines in `%`); not a frame, no tag.

## Verification
Rojas (neuron.pdf): sections 3.1.1, 3.2.1, 3.2.2, 3.3.1, 3.3.2, 4.1, 4.1.1, 4.1.2, 4.1.3, 4.1.4, 4.2,
4.2.1, 4.2.2, 4.2.3, 4.2.4, 4.2.5; figures 3.1, 3.4 (0.9x1 + 2x2 >= 1), 3.5, 3.6, 3.7, 4.3, 4.5, 4.6,
4.8, 4.11; Algorithm 4.2.1 "Perceptron learning" (printed in §4.2, matches the slide's start/test/add/
subtract box). One-step correction formula (delta+eps)/||x||^2 is unnumbered in §4.2.3.
Convergence: Proposition 8 and Eqs. 4.8-4.10 in §4.2.2. Absolute separability: Definition 3, §4.1.3.

## Judgement calls
- "Where Class 2 left us" / "Recap: Last Class": `\nosrc` (recap).
- "A little history": Rojas Fig. 3.1 + §3.1.1 (visible cite §3.1; the history and "only formal
  difference is the weights" are in 3.1.1). Rosenblatt 1958 not tagged: the slide's account is Rojas's,
  the paper is mentioned by name only (see Unverified).
- "The perceptron: a weighted threshold unit": §3.1.1; TikZ unit is the course's own drawing.
- "What does it mean to `learn'?": the TikZ tree reproduces Rojas Fig. 4.3 (learning -> supervised/
  unsupervised -> corrective/reinforcement) -> figure 4.3 + §4.1.1. NB the slide says the perceptron
  rule is "supervised with error correction"; Rojas says the PLA is supervised learning *with
  reinforcement* and its variants are corrective. Text not changed.
- "Extended input space": §4.1.2 (extended vectors) and §4.2 (P, N, open half-spaces).
- "Learning as a search for parameters" (A): §4.1 (parametric system, Fig. 4.1 loop; figure not shown).
  B "Learning as a search in weight space": §4.1 and §4.1.4 (constraints, solution polytope).
- B "The error surface": plot recreates Fig. 4.5 (panels a/b) and the iteration path of Fig. 4.6 (c).
- "Learning as error correction": Algorithm 4.2.1 + §4.2 (informal statement of the same algorithm).
- B "The algorithm running": the course's own run; tagged Algorithm 4.2.1 and §4.2.2 (the result it
  demonstrates: termination on separable data).
- B recreations: weights tilt (regions) -> Fig. 3.4; bias trick TikZ -> Fig. 3.5; duality -> Fig. 3.7;
  update_step -> Fig. 4.8; update_ws -> Fig. 4.11. B "What the weights and the bias each control" is the
  course's own illustration of §3.2.1 (orientation vs offset) -> §3.2.1.
- "The update on the error surface" (A): Fig. 4.11, §4.2.3, plus §4.1.4 (weight-space half-spaces).
- "The perceptron convergence theorem": visible "§4.2-4.3" -> §4.2.2 only (§4.3 is linear programming).
- A "How long can it take? Worst case": §4.2.5 (cited) and §4.2.2. The "inverse squared margin" growth
  is not printed in Rojas; it follows from Eqs. 4.9-4.10 with w0 = 0. Drawing is the course's own.
- "The limit: XOR and non-separable data": §3.2.2 (XOR), §3.3.1 (only linearly separable), §4.2.4 (pocket).
- NPTEL ch6 is named in A's References only ("supporting intuition"); no frame is based on it, no tag.
- Outline, Takeaways, References, Questions?: `\nosrc`.

## Unverified
- Rosenblatt (1958), Psychological Review 65(6) (A and B "A little history", "The perceptron"): paywalled,
  not opened; frames tagged from Rojas only, no `rosenblatt1958` tag.
