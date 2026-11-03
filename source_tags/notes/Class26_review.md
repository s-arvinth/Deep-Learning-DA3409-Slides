# Class 26 review (BPTT and Bidirectional RNNs)

Tags checked: A 36 frames (73 \src, 7 \nosrc); B 41 frames (89 \src, 6 \nosrc). \srcdef: elman1990, jurafsky_slp3 (August 19, 2026 draft text, present in both decks), schuster1997, werbos1990, all consistent with keys.md.

Checked against: J&M aug26 draft: ch. 14 opening ("inherently temporal"), 14.1, 14.1.1 (Eqs 14.1-14.3), 14.1.2 (two-pass training, BPTT, no specialised algorithm, fixed-length segments), 14.2 (Fig 14.5), 14.4.2 (Eqs 14.16-14.18, Fig 14.11), 14.5 (example (14.19)), 14.6 (Fig 14.15).
Goodfellow ch. 10: 10.1 (Eqs 10.1-10.7, Figs 10.1-10.2), 10.2 (Eqs 10.8-10.14, Figs 10.3-10.5, O(tau) time and memory), 10.2.2 (Eqs 10.17-10.28; 10.20/10.21 state recursion, 10.26 recurrent-weight gradient), 10.3 (Fig 10.11), 10.7.
A "Unfolding, drawn" shows only G10_3 (Fig 10.3), so figure 10.3 alone is correct. B's layout covers Figs 10.2-10.3 and is tagged with both.

## Changes
None.

## Checks
check_sources: A 0 errors, 1 warning (References: \nosrc placed last because the body starts with `{\scriptsize`, the guide's known case); B 0 errors 0 warnings. verify_book_numbers: 0 not found. PDFs: A IDENTICAL (53 pages), B IDENTICAL (58 pages).

## Unverified
- "Truncation" (A and B): the \srcline names Williams & Peng (1990). I tried CiteSeerX and MIT Press direct.mit.edu: one returned an empty body and the other timed out. No williams1990 tag added. Still open.
- elman1990 page 182-184, werbos1990 sec. III, schuster1997 sec. II.B: taken on the tagger's word. I tried to re-fetch Elman from papers.baulab.info and it timed out.
