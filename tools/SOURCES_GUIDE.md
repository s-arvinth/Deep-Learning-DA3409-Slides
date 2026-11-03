# How to add source tags (read with tools/SOURCE_REQUIREMENTS.txt)

The brief is `SOURCE_REQUIREMENTS.txt`. This guide settles what it leaves open, so
every class is tagged the same way. Bash paths below; the repo is
`/sessions/compassionate-admiring-einstein/mnt/TA-Duty/experiments/slide-sources`.

## What may change

Only **added lines**: the three `\newcommand` lines, `\srcdef` lines in the
preamble, and `\src` / `\nosrc` lines in frames. No existing character changes,
moves or disappears. `tools/check_sources.py` enforces this against the base commit.
Each rendered PDF must stay pixel-identical to its baseline in
`../slide-sources-work/baseline/Class<NN><A|B>.pdf` (e.g. `Class05A.pdf`).

## Where the lines go

- Preamble: the three `\newcommand` lines exactly as in the brief, then a block of
  `\srcdef` lines, sorted by key, placed just before `\begin{document}` (after any
  existing `\title`/`\footleft` lines is fine). One `\srcdef` per line; a long
  reference may continue on indented lines, but keep it on one line if you can.
- Frames: the tag lines are the **first lines of the frame body**, directly after the
  `\begin{frame}...{Title}` line, indented two spaces, one tag per line, main source
  first. If a tag at the top would change the rendered page (it should not; a frame
  whose body starts with `{` is the one known case), put the tags as the last lines
  before `\end{frame}` instead and say so in the class notes.
- `\maketitle` is not a frame environment and needs nothing. Every
  `\begin{frame}` does, including Outline, section-like standout frames, References
  (`\nosrc`) and the final "Questions?" frame (`\nosrc`).
- Inside `\srcdef{key}{text}` avoid `#` and `%` and keep braces balanced; write `\&`.

## Keys

- The five fixed keys need no `\srcdef`, except `jurafsky_slp3`, which always gets:
  `\srcdef{jurafsky_slp3}{Jurafsky \& Martin, Speech and Language Processing, 3rd edition draft of August 19, 2026, https://web.stanford.edu/~jurafsky/slp3/}`
- Editions (use these PDFs only; their numbers are the ones to cite):
  - `bishop2024` — Bishop & Bishop, Deep Learning: Foundations and Concepts, Springer 2024, 1st ed.
    PDF: `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Christopher M. Bishop, Hugh Bishop - Deep Learning_ Foundations and Concepts-Springer (2024).pdf`
  - `prince2023` — Prince, Understanding Deep Learning, MIT Press 2023, PDF version of October 23, 2023.
    PDF: `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/prince2023udl.pdf`
  - `rojas1996` — Rojas, Neural Networks: A Systematic Introduction, Springer 1996.
    PDF: `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/neuron.pdf`
  - `goodfellow2016` — Goodfellow, Bengio & Courville, Deep Learning, MIT Press 2016.
    PDF: `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/DeepLearningBook_GoodfellowBengioCourville/book/Deep_Learning_Goodfellow_Bengio_Courville.pdf`
  - `jurafsky_slp3` — draft of August 19, 2026.
    PDF: `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/SLP3_JurafskyMartin/book/ed3book_aug26.pdf`
  Search them with `pdftotext -f P -l Q <pdf> -` and `grep`. Every number you write
  must be one you have seen printed in that PDF (section heading, "Figure 6.9",
  equation tag "(6.12)").
- Other sources: `<firstauthorsurname><year>` in lower case, ASCII only (Montúfar →
  `montufar2014`), with a letter suffix when two share a key. The same work gets the
  same key in every deck (check `../slide-sources-work/keys.md`, append to it when
  you create a key, never redefine one differently). `\srcdef` text: authors, title,
  venue, volume/pages or arXiv id, year; for books the edition. The reference
  spreadsheet `/sessions/compassionate-admiring-einstein/mnt/TA-Duty/DA3409_References_by_Class.xlsx`
  (sheet "References by Class") has verified details and links for every work the
  decks cite.
- NPTEL lecture notes (`/sessions/compassionate-admiring-einstein/mnt/TA-Duty/CNS/NPTEL Notes/`)
  are a source like any other: key from the author named in the notes and the year,
  cite `section` or `page` as printed in the notes.

## Deciding what a frame is based on

Read the frame and its existing citations (`\srcline{...}`, `\figsource{...}`,
captions, `\tag{...}`), then confirm in the PDF.

- Cite the place the content comes from, most specific first: figure / equation /
  table / algorithm, then the subsection, then the section, then the chapter. A slide
  that states Eqs. 7.3–7.11 and is built on §7.1.1 gets
  `\src{bishop2024}{equation}{7.3-7.11}` and `\src{bishop2024}{section}{7.1.1}`.
- A figure redrawn from a book figure ("Layout of …, Fig. 14.8", or a B-variant
  recreation of what the A variant shows as a crop) cites that book figure.
- A cropped book figure (A variant, `figs_official/`) cites that figure.
- A theorem, lemma or proposition taken from a paper cites the paper. Use a section
  number if the paper has numbered sections and the result sits in one (most specific:
  the theorem's section); otherwise the page. Get the page from the paper itself:
  fetch the open version with `mcp__workspace__web_fetch` (arXiv `https://arxiv.org/pdf/<id>`,
  JMLR/PMLR/NeurIPS/CVF PDF links from the spreadsheet). Printed page number if the
  paper prints one, else the PDF page. Never guess a page. If you cannot open the
  paper, cite the most specific *verified* place you can (its section, if you can see
  the section structure from an abstract page or a book that quotes it), and record it
  in the class notes under "Unverified".
- `\nosrc` for title-like frames, Outline, recaps, notation slides, takeaways,
  summary tables, References, "Questions?", and the course's own examples,
  derivations and measured experiments that no source contains. A measured
  experiment that tests a book's result: cite the result it tests (that is what the
  slide is based on); the plot itself is the course's own.
- Passing mentions ("as in Fig. 2.4", "see Class 14") get no tag.
- Both A and B variants of a class get tags; frames that are the same in both get the
  same tags.

## Per-class notes

Write `../slide-sources-work/notes/Class<NN>.md` with: number of frames and tags per
variant, the `\srcdef` keys used, every frame where you made a judgement call (one
line why), and an "Unverified" list (frame, tag, what could not be checked). These
notes are not part of the repository.

## Checks before a class counts as done

```bash
cd /sessions/compassionate-admiring-einstein/mnt/TA-Duty/experiments/slide-sources
python3 tools/check_sources.py ClassNN_*/A_*/main.tex ClassNN_*/B_*/main.tex     # 0 errors, read the warnings
tools/build_deck.sh ClassNN_*/A_Textbook_Figures /tmp/ClassNNA.pdf
python3 tools/compare_pdfs.py ../slide-sources-work/baseline/ClassNNA.pdf /tmp/ClassNNA.pdf   # IDENTICAL
(same for B)
```
Do not commit; the coordinator commits after an independent review.
