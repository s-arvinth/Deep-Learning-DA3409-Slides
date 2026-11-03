# Source tags: report (branch exp/slide-sources)

Every frame of the 42 decks (Classes 2–29, A and B variants, including the Class 29 A
variant added on this branch) carries `\src{source}{type}{number}` lines or `\nosrc`,
following `SOURCE_REQUIREMENTS.txt`. Each deck's preamble has the brief's three no-op
macros and one `\srcdef` per non-fixed key. How the brief was applied is in
`SOURCES_GUIDE.md`; the shared key list is `keys.md`; per-class working notes are in
`notes/` (`ClassNN.md` from the tagging pass, `ClassNN_review.md` from the independent
review).

## Totals

| | |
|---|---|
| Decks | 42 (21 classes × A, B) |
| Frames | 1,387, every one tagged |
| `\src` tags | 2,440: section 1,487, figure 473, equation 398, page 37, algorithm 19, chapter 18, table 8 |
| `\nosrc` frames | 352 (title-like, outline, recap, notation, takeaways, summary tables, references, the course's own derivations) |
| Keys | 5 fixed + 97 `\srcdef` keys, identical text in every deck that uses them |

## How it was checked

1. **Only lines added.** `git diff 03bf6e7 -- Class*`: 3,173 lines added, 0 removed or changed.
2. **Slides unchanged.** Every deck rebuilt and compared page by page with its pre-change
   build at 50 dpi (`tools/compare_pdfs.py`): 42 of 42 pixel-identical.
3. **Format.** `tools/check_sources.py all`: 0 errors in 42 decks (every frame tagged,
   keys defined, allowed types, bare numbers, no tag in a comment). One warning: the Class 26
   A References frame has its `\nosrc` last, because its body starts with `{`.
4. **Book numbers exist.** `tools/verify_book_numbers.py`: all 2,034 tags into the five books
   name a figure caption, equation tag, section heading, table or algorithm printed in the
   edition used (Bishop & Bishop 2024; Prince PDF of 23 Oct 2023; Rojas 1996; Goodfellow et
   al. 2016; Jurafsky & Martin draft of 19 Aug 2026).
5. **Right place.** Each class was tagged by one worker and audited by a second, who read the
   cited figure, equation or section against the slide and fixed ~45 tags (missing equation
   tags in end-of-chapter notes and exercises, two wrong sections, less specific tags, one
   over-tag, incomplete `\srcdef` entries).

## For the author to check

Places the workers could not open, so the tag rests on a secondary copy, or no tag was added:

- **Nesterov (2004)**, Class 13 (9 tags, section 1.2.2) and Class 14 (chapter 2): the 2004 edition
  could not be opened. From the 1998 draft, the one-step decrease and step-size discussion may
  belong in section 1.2.3 rather than 1.2.2.
- **Whole-article page ranges** (page of the result not found): `wolpert1996` 1341-1390 and
  `nowlan1992` 473-493 (Class 18).
- **Read in a version other than the cited one:** `polyak1964` (Russian original), `bishop1995a`
  (tech report), `bishop1995b` (postprint), `breiman1996` (tech report), `williams1989`
  (authors' version), `bishop2006` §3.5.3 (table of contents only).
- **Section numbers resting on one reading:** `hornik1991` §2, `barron1993` §III–IV,
  `devore1989` §1, §4 (Classes 5, 7, 9).
- **Named on a slide but not tagged** (paper not opened): Rosenblatt 1958 (Class 3),
  Zaslavsky 1975 (Class 5), Hyvärinen et al. 2009 (Class 25), Williams & Peng 1990 (Class 26).
  The slide content is tagged at the book section that presents it.
- **`chakravarthy2012`** (NPTEL notes): author and year taken from the PDF metadata.
- **Attribution on a slide that the source does not support:** Class 23 "Convolution is the
  equivariant linear map" credits the converse to Cohen & Welling (2016); that paper proves only
  the forward direction.

## Visible text on the slides that disagrees with the editions

Found while tagging; the slides were not changed. The tags follow the books.

- Class 5 A: Prince figures captioned 2.2, 3.12, 3.11 are Figs. 2.1, 3.11, 3.12.
- Classes 5–6: "Bishop Eqs. 6.1–6.3, §6.1" is Eqs. 6.7–6.9, §6.2; "§6.1.1" (weight-space
  symmetry) is §6.2.4; "§4.1.1" (least squares from Gaussian noise) is §4.1.2.
- Class 6 B: a commented-out `\tag{B\&B 5.7}` should be 5.5.
- Class 9 References: DeVore et al. (1989) is in Manuscripta Mathematica, not Numer. Math.
- Class 10: "Fig. 4.9" (regions vs parameters) is Prince Fig. 4.7.
- Class 11: Prince "Fig. 5.14" is 5.5 and "Fig. 7.8" is 7.9; `\tag{B&B 8.3–8.5}` is Eqs. 8.7–8.10;
  "Which mode, and why" labels Bishop Fig. 8.5 as reverse mode, but Figs. 8.4–8.5 are both
  forward-mode traces.
- Class 25: the 1×1 convolution slide cites Prince §10.5.1; the text is §10.4.3 and Fig. 10.14.

## Reproduce the checks

```bash
python3 tools/check_sources.py all
python3 tools/verify_book_numbers.py        # needs the five book PDFs at the paths in SOURCES_GUIDE.md
tools/build_deck.sh Class14_Convergence/A_Textbook_Figures /tmp/14A.pdf
python3 tools/compare_pdfs.py <pre-change build>.pdf /tmp/14A.pdf
```
