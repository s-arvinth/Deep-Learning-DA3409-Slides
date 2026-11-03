# Class 10 review (Depth and Width Efficiency, Theorems)

Reviewer: independent check of the tagger's tags (A: 29 frames, 33 \src + 10 \nosrc; B: 29 frames, 34 \src + 10 \nosrc).
All 43 + 44 tag lines and 11 \srcdef lines checked.

## How checked
- Papers fetched from arXiv and section headings / theorem positions read:
  Telgarsky 1602.04485 (Thm 1.1 in 1.1; Lemmas 3.10/3.11 in 3.3), Eldan-Shamir 1512.03965 (Thm 1 in sec 1),
  Liang-Srikant 1610.04161 (3.1 upper bounds, 4 lower bounds), Cohen et al. 1509.05009 (sec 4 Theorems of network capacity),
  Safran-Shamir 1610.09887 (Thms 1-2 in sec 3), Lu et al. 1709.02540 (sec 4 Width efficiency vs depth efficiency),
  Vardi et al. 2202.03841 (sec 3), Montufar et al. 1402.1869 (Thm 4 in 3.2 Formal Result),
  Serra et al. 1711.02114 (Thm 1 in 3.1, exact counting sec 5), Bartlett et al. 1703.02930 (Eq. (2) and Thms 3, 6 in 1.1),
  Poggio et al. 1611.00740 (Thm 1 in 3.3, Thm 2 in 3.4).
- Prince: printed p. 53 holds the chapter-4 Notes "Depth efficiency" / "Width efficiency" (and Eq. 4.17); Figure 4.7 is the
  regions-vs-parameters figure (Di=1, Di=10) in 4.5.2; Figure 4.9 is the Problem 4.8 figure (tagger right).
- Bishop 6.3.1 Hierarchical representations: edges -> parts -> objects, compositional inductive bias. Correct.
- Prince Eq. 4.17 checked against "Counting linear regions": it is the (floor(D/Di)+1)^Di form (Serra's), not the slide's
  floor(M/D)^D Montufar form, so not added (agrees with tagger).

## Changes
1. srcdef (A and B, and keys.md): poggio2017 "International Journal of Automation and Computing 14:503" ->
   "International Journal of Automation and Computing 14(5), 503-519" (volume/issue/pages from Crossref,
   DOI 10.1007/s11633-017-1054-2). Resolves the tagger's only open point.

No wrong-place, missing, less-specific or over-tagged findings.

## Checks
check_sources: A 0 errors 0 warnings, B 0 errors 0 warnings; verify_book_numbers: 0 not found;
build + compare_pdfs: A IDENTICAL (39 pages), B IDENTICAL (39 pages).

## Still unverified
- none.
