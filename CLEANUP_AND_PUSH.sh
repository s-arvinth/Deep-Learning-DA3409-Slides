#!/bin/bash
# One-off tidy: Class 6's recreated-figures variant no longer references
# the textbook images that were copied in while it was a scaffold.
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$REPO/Class06_Shallow_NN_Classification/B_Recreated_Figures/figs_official"
echo "Removed the unused figs_official copy from Class 6 B."
echo
echo "Now push with:"
echo "  cd ~/Documents/TA-Duty && ./overleaf-sync.sh push \"Class 6 recreated figures\""
