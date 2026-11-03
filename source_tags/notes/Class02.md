# Class 02 — Introduction to Neural Networks

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 23 | 33 | 4 | 0 errors, 0 warnings | IDENTICAL (29 pages) |
| B_Recreated_Figures | 23 | 32 | 3 | 0 errors, 0 warnings | IDENTICAL (29 pages) |

`\srcdef` keys: `chakravarthy2012` (A, B; NEW, appended to keys.md), `oeis2024` (B; existing key).
Fixed key used: `rojas1996`. All tags are the first lines of the frame body (no body starts with `{`).
B has a commented-out "Questions?" frame (lines in `%`); it is not a frame and gets no tag.

## Verification
- Rojas (neuron.pdf, pdftotext): sections 1.1.1, 1.2.1, 1.2.2, 1.2.3, 1.3.1, 1.3.2, 2.1.1, 2.1.2, 2.2.1,
  2.2.2, 2.2.3, 3.2.2, 3.3.1; figures 1.14, 1.16, 2.6, 2.7, 2.8, 2.12, 2.13, 3.6 (all captions seen).
  Section of a figure = section whose text it sits in.
- NPTEL Computational Neuroscience notes (CNS/NPTEL Notes): section headings 1.1.5.2 (Summation at the
  soma and Action Potential; all-or-none, threshold, refractory period), 1.1.5.3 (Axonal propagation;
  AP propagates without losing amplitude), 5.4.2 (Leaky integrate-and-fire; thresholding, firing rate).
  Section numbers are unique across the chapter PDFs, so one key covers the set.
- OEIS A000609 fetched (oeis.org): 2, 4, 14, 104, 1882, 94572, ... matches the slide's 4, 14, 104, 1882,
  94572 for n=1..5. Cited as `\src{oeis2024}{page}{A000609}` as keys.md prescribes.

## Judgement calls
- "The brain as an inspiration": tagged Rojas 1.3.1 (the visible cite says §1.3; "networks of primitive
  functions" is in 1.3.1) and 1.1.1 (ms switching, massive parallelism). B keeps 1.1.1 because the
  "massively parallel" bullet is still visible (only the ms bullet is commented out).
- "Anatomy of a neuron": visible cite §1.2 -> 1.2.1 (dendrites/soma/axon/synapses) and 1.2.3
  (threshold firing, excitatory/inhibitory). TikZ neuron is the course's own schematic, no figure tag.
- A "Firing: an all-or-nothing threshold": visible "cf. NPTEL ... simple neuron models" -> NPTEL 1.1.5.2
  (main: all-or-none, threshold) and 5.4.2 (simple neuron model, thresholding, rate); Rojas 1.2.2 added
  (all-or-nothing signal of same form and amplitude). B's merged firing slide: NPTEL 1.1.5.2, Rojas 1.2.2.
- A "How the signal travels": Rojas 1.2.2 (cited) plus NPTEL 1.1.5.3 (no loss of strength) and 1.1.5.2
  (refractory period; not in Rojas). The spike drawing is a generic schematic, not tagged as a figure.
- B "Simple units add up to complicated functions": the staircase fit is the course's own experiment;
  tagged Rojas 1.3.2 (networks as function approximators), the idea it illustrates (A cites 1.3.2 there).
- "Threshold logic is expressive": visible §2.2 -> 2.2.1 (n-ary AND/OR, NOT) and 2.2.3 (any Boolean
  function). "Any Boolean function, by construction": 2.2.3 only (Fig. 2.14 is described, not shown).
- B recreations cite the book figure the A crop shows: abstract neuron -> 1.14, M-P unit -> 2.6, step
  plot (a) -> 2.7, AND/OR gates -> 2.8, cube_or -> 2.12, cube_majority -> 2.13.
- B "The same gates, drawn in input space": AND/OR split by a line in 2-D input space, XOR not -> Rojas
  Fig. 3.6 and §3.2.2 (Ch. 3, not Ch. 2). The spreadsheet lists OEIS for this frame and for
  "Threshold logic is expressive"/"Any Boolean function"; not followed, OEIS counts appear only on
  "How rare is linear separability?".
- "The one it cannot do: XOR" -> Rojas 3.2.2; "Linear separability --- and its limit" -> 3.3.1 (also the
  source of "most functions of many variables are not": 14/16, 104/256, 1882/65536).
- "How rare is linear separability?": OEIS first (n=1..5 counts in the plot), Rojas 3.3.1 second.
- Outline, Takeaways, References, Questions?: `\nosrc`.

## Unverified
- `chakravarthy2012` (A and B, all NPTEL tags): the notes print no author or year. Author (V. S.
  Chakravarthy, IIT Madras) and year (2012) are inferred from the PDF metadata (Author "VSC" /
  "Srinivas", CreationDate Dec 2012). Section numbers themselves were seen printed.
- "The brain as an inspiration" (A and B): the figures 10^11 neurons / 10^4 connections are not in
  Rojas; NPTEL ch0_introv2 says "about 100 billion neurons, each with about 1,000-10,000 connections"
  on its PDF page 2, but that intro chapter has no section number and no printed page number, so no
  tag was added for it.
