# Class 23 review (independent check of the source tags)

## Checked
- A: 26 frames, 49 \src + 9 \nosrc (before review), 3 \srcdef. B: 32 frames, 68 \src + 8 \nosrc, 3 \srcdef.
- All Bishop 2024 (§10.1.1, 10.2, 10.2.1-10.2.6; Eq. 10.1; Figs 10.2, 10.4) and Prince 2023 (§10.1-10.4.2;
  Eqs 10.1-10.6; Figs 10.1-10.6, 10.9-10.12) tags read against the book text; correct.
- cohen2016a (arXiv 1602.07576): §5 "Equivariance properties of CNNs" shows the forward direction only.
- simonyan2015 (arXiv 1409.1556): §2.3 "Discussion" gives the 3x3-stack argument (two 3x3 = 5x5 field, three = 7x7,
  27C^2 vs 49C^2, three nonlinearities instead of one).
- After edits: check_sources 0 errors / 0 warnings, verify_book_numbers 0 not found, PDFs IDENTICAL (A 38 pages,
  B 40 pages).

## Changes
1. "Three knobs, and one formula" (A, B): added \src{bishop2024}{equation}{10.5} before the Bishop sections (less
   specific: the strided output-size formula is printed as Eq. 10.5 in §10.2.4; only the sections were cited).
2. A "Why depth beats a wider kernel": \nosrc -> \src{simonyan2015}{section}{2.3} (missing: the table and bullets are
   VGG's argument, including the 27C^2 / 49C^2 counts; not the course's own).
3. A preamble: added \srcdef{simonyan2015}{...} with the keys.md text (srcdef; needed by change 2), sorted after luo2016.
Totals: less specific 2, missing 1, srcdef 1, wrong place 0, over-tagged 0.

## Still unverified
1. "Convolution is the equivariant linear map" -- \src{cohen2016a}{section}{5}: §5 has the forward statement only;
   the converse that the slide attributes to Cohen & Welling (2016) is not in that paper (searched the arXiv text).
   Tag kept because the slide names that source; the attribution itself should be checked by the author.
