# Class 27 review (independent reviewer)

Tags checked: A 33 frames (80 \src + 7 \nosrc), B 37 frames (87 \src + 7 \nosrc), 9 \srcdef per deck.
Jurafsky numbers checked in ed3book_aug26.pdf (draft of August 19, 2026; pdftotext -layout of PDF pp. 285-338):
Eqs. 14.4-14.9 in 14.2.1, Fig. 14.5 in 14.2, Fig. 14.9 in 14.3.3, Fig. 14.16-14.19 and Eqs. 14.28-14.33 in 14.7/14.7.1,
Fig. 14.20-14.22 and Eqs. 14.34-14.38 in 14.8 (14.38 is the bilinear W_s score), Eqs. 13.14-13.16 and Fig. 13.7 in 13.4,
13.4.1 (MBR), 13.2.2 (parallel training data), 13.1.1 (example 13.4 bruja verde). Goodfellow (book PDF pp. 388-436):
Eqs. 10.15-10.16 in 10.2.1, Eq. 10.31 and 10.34 in 10.2.3 (end symbol, Bernoulli stop, predict tau), Fig. 10.9 and the
"extra input / initial state / both" list in 10.2.4, Fig. 10.12 and the too-small-context paragraph in 10.4.
Papers: cho2014 2.2 re-opened (arXiv 1406.1078v3); williams1989 2.3 "Teacher-Forced Real-Time Recurrent Learning"
re-opened (authors' copy, "Appears in Neural Computation 1, pp. 270-280"). Other paper sections accepted as recorded by the tagger.

All frame tags found correct, complete and specific; A and B agree on shared frames; B-only experiment frames cite
the results they test.

## Changes (frame, old -> new, reason)
srcdef (6, both variants; keys.md rows updated identically; keys only used in Classes 27-29):
- bahdanau2015: "ICLR 2015, 2015" -> "ICLR 2015; arXiv:1409.0473, 2015" (no volume/pages or arXiv id)
- bengio2015: "NeurIPS 2015, 2015" -> "NeurIPS 2015; arXiv:1506.03099, 2015"
- cho2014: "EMNLP 2014, 2014" -> "EMNLP 2014; arXiv:1406.1078, 2014"
- luong2015: "EMNLP 2015, 2015" -> "EMNLP 2015; arXiv:1508.04025, 2015"
- ranzato2016: "ICLR 2016, 2016" -> "ICLR 2016; arXiv:1511.06732, 2016"
- sutskever2014: "NeurIPS 2014, 2014" -> "NeurIPS 2014; arXiv:1409.3215, 2014"
(arXiv ids from the reference spreadsheet / keys.md links; no pages added because none were verified.)

No frame tag changed.

## Unverified / left as is
- shannon1951: not tagged. J 14.3.3 names Shannon (1951) only as the origin of the sampling idea; the slide
  ("Generating, one token at a time") is J Fig. 14.9. A passing attribution, so no tag even if a copy were opened.
- williams1989 2.3: checked on the authors' version, not the paywalled journal PDF.

Checks: check_sources 0 errors (A, B); verify_book_numbers 0 not found; builds IDENTICAL to baseline (48 / 53 pages).
