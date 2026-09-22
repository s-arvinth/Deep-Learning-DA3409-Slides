# Class 27 — C variant (textbook theorems only)

The B deck with the three results that appear in neither set text removed.

Removed:

| Slide | Rests on |
|---|---|
| Proposition *Exposure bias* | a formal statement of the train/test mismatch Goodfellow §10.2.1 and Jurafsky §14.7.1 describe only in words |
| Proposition *The context is a convex combination* and its proof | elementary, but in neither book |
| Proposition *Cost and optimality of beam search* and its proof | the O(k m \|V\|) bound and the non-optimality statement, which Jurafsky §13.4 describes without stating |

Rewritten: each of the three slides keeps the books' own remarks in the
same place, and the measured figures survive with captions confined to
what the books say.

Kept, because the books state them: the RNN language model (Jurafsky
Eqs. 14.4–14.7); the chain rule (Jurafsky Eqs. 14.8–14.9, 14.28; Goodfellow
Eq. 10.31); the encoder–decoder and its equations (Jurafsky Eqs.
14.32–14.33); teacher forcing as maximum likelihood (Goodfellow Eqs.
10.15–10.16); dot-product and bilinear attention (Jurafsky Eqs.
14.34–14.37); greedy and beam decoding (Jurafsky §13.4).

49 pages against the B variant's 53. Compile with `xelatex main.tex`, twice.
