#!/bin/bash
# =====================================================================
#  Finish the reorganisation, then push to Overleaf.
#  Run once from the repo root:
#      cd ~/Documents/TA-Duty/DeepLearning-Overleaf && bash CLEANUP_AND_PUSH.sh
# =====================================================================
set -e
cd "$(dirname "$0")"

echo "==> removing the old flat copies (now living in A_/B_ subfolders)"
for c in Class02_Intro_Neural_Networks Class03_Perceptrons_PLA \
         Class05_Shallow_NN_Regression Class06_Shallow_NN_Classification \
         Class07_Deep_Networks_Universal_Approximation; do
  rm -f  "$c"/main.tex "$c"/main.pdf "$c"/main.aux "$c"/main.log \
         "$c"/main.nav "$c"/main.out "$c"/main.snm "$c"/main.toc
  rm -rf "$c"/figs "$c"/figs_official "$c"/fonts "$c"/src
done

echo "==> removing the superseded v2 folders"
rm -rf Class02v2_Intro_Neural_Networks \
       Class03v2_Perceptrons_PLA \
       Class07v2_Deep_Networks_Universal_Approximation

echo "==> removing staging and stray build artefacts"
rm -rf _staging
find . -name "main.aux" -o -name "main.log" -o -name "main.nav" \
     -o -name "main.out" -o -name "main.snm" -o -name "main.toc" | xargs rm -f 2>/dev/null || true
find . -name "*.png" -path "*/figs/*" | xargs rm -f 2>/dev/null || true   # superseded rasters
rm -f Class07_Deep_Networks_Universal_Approximation/B_Recreated_Figures/figs/own_*.pdf 2>/dev/null || true

echo "==> resulting structure"
find . -maxdepth 2 -type d -name "[AB]_*" | sort

echo "==> committing and pushing"
git add -A
git commit -m "Reorganise into per-class A_Textbook_Figures / B_Recreated_Figures variants; update Class 2, 3, 7 recreated decks"
git push
echo "Done."
