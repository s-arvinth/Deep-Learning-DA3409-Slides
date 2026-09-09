#!/bin/bash
# Thin wrapper so the helper can be invoked from inside the repo too.
# The real script lives one level up, next to the clone.
exec "$(cd "$(dirname "$0")/.." && pwd)/overleaf-sync.sh" "$@"
