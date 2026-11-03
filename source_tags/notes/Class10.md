# Class 10 — Depth and Width Efficiency, Theorems

| variant | frames | \src | \nosrc | check | PDF vs baseline |
|---|---|---|---|---|---|
| A_Textbook_Figures | 29 | 33 | 10 | 0 errors, 0 warnings | IDENTICAL (39 pages) |
| B_Recreated_Figures | 29 | 34 | 10 | 0 errors, 0 warnings | IDENTICAL (39 pages) |

Tags are the first lines of every frame body (no frame body starts with `{`).

## \srcdef keys used (both variants)
bartlett2019 (new), cohen2016b, eldan2016, liang2017, lu2017, montufar2014, poggio2017 (new),
safran2017, serra2018 (new), telgarsky2016, vardi2022 (new). Fixed keys: prince2023, bishop2024.
New keys appended to keys.md.

## How the numbers were checked
- Papers: arXiv PDFs fetched (Telgarsky 1602.04485, Eldan-Shamir 1512.03965, Liang-Srikant 1610.04161,
  Cohen et al. 1509.05009, Safran-Shamir 1610.09887, Lu et al. 1709.02540, Vardi et al. 2202.03841,
  Montufar et al. 1402.1869, Serra et al. 1711.02114, Bartlett et al. 1703.02930, Poggio et al. 1611.00740).
  Every theorem is cited by the numbered section that contains it:
  Telgarsky Thm 1.1 = sec 1.1, sawtooth Lemmas 3.10/3.11 = sec 3.3; Eldan-Shamir Thm 1 = sec 1;
  Liang-Srikant deep upper bounds sec 3.1, lower bounds/comparison sec 4; Cohen Thm 1 = sec 4;
  Safran-Shamir Thms 1-2 = sec 3; Lu Thm 4 = sec 4; Vardi Thm 3.1 = sec 3; Montufar Thm 4 = sec 3.2;
  Serra upper bound Thm 1 = sec 3.1, exact counting = sec 5; Bartlett Thms 3, 6 and Eq. (2) = sec 1.1;
  Poggio Thm 1 (shallow) = sec 3.3, Thm 2 (deep, compositional) = sec 3.4.
- Prince (PDF of Oct 23, 2023): sec 4.5.2, Figure 4.7, chapter Notes on printed page 53.
- Bishop: sec 6.3.1 Hierarchical representations.

## Judgement calls
- "Regions against parameters" (A): the slide caption says "Prince (2023), Fig. 4.9", but in the
  edition we cite Figure 4.9 is the Problem 4.8 figure; the regions-vs-parameters figure (Di=1 and
  Di=10 panels, as described in the caption) is **Figure 4.7** (sec 4.5.2). Tagged figure 4.7; the
  visible caption was not changed. The coordinator may want to fix the visible "Fig. 4.9" text.
- "Regions against parameters" (B): the course's own plot of the Montufar lower bound and the product
  upper bound; tagged montufar2014 3.2, serra2018 3.1, then prince2023 4.5.2 (the section the A
  variant's figure comes from; same quantity plotted).
- Theorem wording on "Depth beats width", "Not just one contrived function", "The question, turned
  around", "Width efficiency" and the highlight of "The asymmetry, and what it decides" follows
  Prince's chapter-4 Notes ("Depth efficiency", "Width efficiency") nearly verbatim; the Notes are an
  unnumbered section, so they are cited as prince2023 page 53, after the paper.
- "The asymmetry, and what it decides" is a summary table but its conclusion is Prince p. 53's
  sentence, so it gets \src{prince2023}{page}{53} rather than \nosrc.
- "Today's question, in one line": a roadmap table of who answers what -> \nosrc (outline-like).
- "The separation, measured" / "The cost of flattening" / "VC dimension, drawn" / "The size of the
  effect": course-generated plots; tagged with the result each one plots (Telgarsky; Telgarsky +
  Lu + Vardi; Bartlett Eq. 2; Poggio).
- "Why real data is compositional": Bishop 6.3.1 (edges -> parts -> objects, compositional
  inductive bias). Poggio et al. have a "Why are compositional functions so common?" part, but its
  "7" is ambiguous (item list inside sec 6 vs a section), so it was not cited.
- "Counting linear regions": formula is Montufar Thm 4 (not Prince Eq. 4.17, which has a different
  form), so only montufar2014 3.2.
- Recaps / outline / summary table / "What none of them say" / takeaways / references / questions: \nosrc.

## Unverified
- none. (All section numbers seen in the fetched PDFs; poggio2017 \srcdef uses the volume and first
  page printed on the deck's References slide, "14:503", not checked against the journal.)
