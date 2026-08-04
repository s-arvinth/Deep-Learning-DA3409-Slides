#!/usr/bin/env python3
"""Deprecated entry point -- kept so old commands keep working.

The figures now live one per file under figures/, with the shared style,
data and models in common/.  Use:

    python3 build_all.py                  # everything
    python3 build_all.py sawtooth         # just the matching figure
    python3 figures/fig05_sawtooth.py     # or run the file directly
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_all import main

if __name__ == "__main__":
    main(sys.argv[1:])
