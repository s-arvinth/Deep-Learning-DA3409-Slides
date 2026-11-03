#!/bin/bash
# Compile one deck folder twice with XeLaTeX in a scratch copy (theme files copied
# in from the repo root) and copy the PDF to $2. Prints page and err/vbox/hbox counts.
# Usage: tools/build_deck.sh <deck-folder> <out.pdf>
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D="$(cd "$1" && pwd)"; OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
T="$(mktemp -d)"; cp -r "$D"/. "$T"/; cp "$REPO"/beamer*metropoliscustom.sty "$T"/
[ -d "$T/fonts" ] || cp -r "$REPO/fonts" "$T/fonts"
# decks that name images by repo-root paths (Overleaf compiles from the root)
REL="${D#$REPO/}"; mkdir -p "$T/$REL"; for f in figs figs_official; do [ -d "$T/$f" ] && ln -s "$T/$f" "$T/$REL/$f"; done
cd "$T"
for i in 1 2; do timeout 150 xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1; done
cp main.pdf "$OUT" 2>/dev/null
echo "$(basename "$OUT") pages=$(pdfinfo "$OUT" 2>/dev/null | awk '/Pages/{print $2}') err=$(grep -c '^!' main.log) vbox=$(grep -c 'Overfull .vbox' main.log) hbox=$(grep -c 'Overfull .hbox' main.log)"
rm -rf "$T"
