# Class 24 — Applications: Detection, Segmentation

| variant | frames | \src | \nosrc |
|---|---|---|---|
| A_Textbook_Figures | 39 | 67 | 8 |
| B_Recreated_Figures | 38 | 70 | 7 |

Checks: `tools/check_sources.py` reports 17 (A: 18) ERRORs "malformed \src (needs three arguments)" on every frame
that contains `\srcline{...}` — a false positive of the checker (the regex `\\src(?!\{)` matches the
deck's own `\srcline` macro; only `\srcdef` is excluded). With that one regex also excluding `\srcline`
(copy in /tmp, tool itself not modified) both decks give 0 errors, 0 warnings. PDFs: both IDENTICAL to baseline
(A 54 pages, B 45 pages). No frame body starts with `{`, so all tags are at the top of the frame.

## \srcdef keys
A: badrinarayanan2017, dumoulin2016, girshick2015, krizhevsky2012, long2015, redmon2016, ren2015, sermanet2014, simonyan2015
B: badrinarayanan2017, dumoulin2016, krizhevsky2012, long2015, redmon2016, sermanet2014, simonyan2015
(all existing keys from keys.md; no new key created)

## Where the numbers were checked
- Bishop ch. 10: 10.1, 10.4, 10.4.1–10.4.6, 10.5, 10.5.1–10.5.4 headings; Figs 10.19–10.31 captions; Fig 10.27 is referenced in 10.5.2 text, Fig 10.30 sits in 10.5.3.
- Prince ch. 10: 10.4.2, 10.5.1–10.5.3; Figs 10.13, 10.15–10.21 (10.21 is in the chapter Notes, no section, so figure only). All AlexNet/VGG/ImageNet/YOLO numbers on the slides are printed in 10.5.1/10.5.2.
- Papers (arXiv/NeurIPS PDFs fetched): sermanet2014 §3.5 "ConvNets and Sliding Window Efficiency"; redmon2016 §2 "Unified Detection" (S×S×(B*5+C)), §2.2 Training, Eq. (3) loss with λcoord=5, λnoobj=.5; girshick2015 §2 "Fast R-CNN architecture and training"; ren2015 §3.1 "Region Proposal Networks"; long2015 §3 "Fully convolutional networks" (any input size); badrinarayanan2017 §3 "Architecture" (max-pooling indices; arXiv v3 numbering); dumoulin2016 §4.1 "Convolution as a matrix operation", §4.2 "Transposed convolution"; krizhevsky2012 §3.5 "Overall Architecture", Table 2 (5 CNNs: 38.1 / 16.4); simonyan2015 §2.2 "Configurations" (Table 2, 144M), Table 7 (23.7 / 6.8).

## Judgement calls
- Outline, "From the layer to the task", "Notation for today", "Proof of the proposition" (A; own proof), "Three tasks, one trunk", Takeaways, References, Questions: \nosrc. Notation has a \srcline but the guide makes notation slides \nosrc.
- Frames that only state a result whose book figure is drawn on the *next* frame (Scoring a box, Sliding windows, A sliding window is a convolution, Non-max suppression, Predicting boxes directly, Undoing pooling, Learned up-sampling (A)) cite the section only; the figure is cited on the frame that shows it.
- Bishop "Exercise 10.12/10.13" in \srcline: not a citable type; covered by the section tag.
- Captions "adapted from Russakovsky et al. / Redmon et al. / Noh et al." on Prince figures: attribution of the book figure, not tagged (the figure frame cites the Prince figure).
- "Two networks, one lesson": main Prince §10.5.1, plus the papers' places where the table's numbers appear (krizhevsky2012 Table 2, §3.5; simonyan2015 Table 7, §2.2).
- B "What the output structure decides": own figure with the books' numbers; tagged Prince §10.5.1, 10.5.2, 10.5.3 (its own caption cites §10.5.1–10.5.3).
- B "IoU, drawn and measured" / "The saving, drawn" / "Unpooling, drawn" / "Transpose convolution, drawn": recreations, cite the Bishop figures they redraw plus section.
- B "Object Detection and Objective Function": the YOLO loss, redmon2016 Eq. 3 and §2.2.
- B "Learned up-sampling": the uncaptioned image figs/Image_Report.png is Prince Fig 10.13 (transposed convolution in 1D, panels a–d), in §10.4.2; tagged after the Bishop §10.5.3 main source.

## Unverified
(none)
