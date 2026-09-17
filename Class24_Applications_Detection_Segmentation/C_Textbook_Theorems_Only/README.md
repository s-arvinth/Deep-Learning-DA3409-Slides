# Class 24 — C variant (textbook theorems only)

The B deck with the two results that appear in neither set text removed.

Removed:

| Slide | Rests on |
|---|---|
| Proposition *What the ratio buys* and its proof | Bishop §10.4.2 states only that IoU lies in [0, 1]; the coincidence and scale-invariance properties are elementary but not in either book |
| Proposition *A bottleneck alone cannot place every pixel* | a rank argument about a linear decoder, stated in neither book |

Rewritten: *Scoring a box* keeps the definition and Bishop's own remarks
(range, the 0.5 threshold, why a ratio rather than an area, evaluation
rather than training); *What the bottleneck discards* keeps the measured
figure with a caption confined to what Bishop §10.5.4 says about lost
spatial information.

Kept, because the books state them: the bounding-box parameterisation
(Bishop §10.4.1); sliding windows as a single convolutional pass (Bishop
§10.4.3, Figs. 10.22–10.23, Exercise 10.12); non-max suppression (Bishop
§10.4.5); the YOLO output tensor (Prince §10.5.2); unpooling inverting
pooling (Bishop §10.5.2); transpose convolution as the transpose matrix
(Bishop §10.5.3, Exercise 10.13); fully convolutional networks accepting any
size (Bishop §10.5.3).

52 pages against the B variant's 55. Compile with `xelatex main.tex`, twice.
