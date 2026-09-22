"""The one experiment the deck's measured figures share.

Two encoder-decoders of the same size (d = 48) are trained on the
reversal task for the same number of steps on sources of length 2-10:
one with the fixed context c = h^e_n, one with dot-product attention.
Both are then tested on lengths 2-16, beyond what they saw.  Results
are cached in ../figs/_cache.pkl so the figures agree with each other.
"""
import os, pickle
import numpy as np
from . import models as M

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figs", "_cache.pkl")
STEPS = 4000
LENS_TEST = list(range(2, 17))


def run(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    out = {}
    for name, att in (("fixed", False), ("attention", True)):
        print("training the %s-context model" % name)
        rng = np.random.default_rng(1)
        m = M.Seq2Seq(d=48, attention=att, seed=2)
        hist = M.train(m, STEPS, rng, log_every=500)
        acc = M.exact_match(m, np.random.default_rng(7), LENS_TEST)
        print("  exact match by length:", {k: round(v, 2) for k, v in acc.items()})
        out[name] = dict(model=m, hist=hist, acc=acc)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out
