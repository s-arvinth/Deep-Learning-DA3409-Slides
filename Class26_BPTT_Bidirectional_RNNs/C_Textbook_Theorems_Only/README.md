# Class 26 — C variant (textbook theorems only)

The B deck with the one result that appears in neither set text removed.

Removed:

| Slide | Rests on |
|---|---|
| Proposition *What truncated BPTT removes* and its proof | the statement that cutting the backward pass after k steps drops every term of the parameter gradient from more than k steps back; Jurafsky §14.1.2 describes segmenting a long input in words, and the result is Williams & Peng (1990) |

Rewritten: the slide keeps the book's own description of fixed-length
segments in the same place, and the measured figure survives with a
caption confined to what the book says. The takeaway is reworded.

Kept, because the books state them: the simple recurrent network
(Jurafsky Eqs. 14.1–14.3, Goodfellow Eqs. 10.8–10.11); unfolding
(Goodfellow Eqs. 10.1–10.7); back-propagation through time (Goodfellow
Eqs. 10.17–10.28); the product of Jacobians (Goodfellow Eq. 10.21
iterated, §10.7); the O(τ) cost (Goodfellow §10.2); the bidirectional RNN
and what it depends on (Jurafsky Eqs. 14.16–14.18, Goodfellow §10.3).

57 pages against the B variant's 59. Compile with `xelatex main.tex`, twice.
