"""The one experiment the deck's measured figures share (Class 27).

Ported from Class27_Encoder_Decoder_Seq2Seq/B_Recreated_Figures/src/common/
experiment.py, same seeds and settings: two encoder-decoders of the same
size (d = 48) are trained on the reversal task for 4,000 steps on sources
of length 2-10, one with the fixed context c = h^e_n, one with dot-product
attention; both are then tested on lengths 2-16.

Cached stage by stage in output/_cache_<stage>.pkl (one stage per model);
training also writes output/_ckpt_<stage>.pkl every 250 steps so an
interrupted run resumes exactly where it stopped.  Delete output/ to
recompute from scratch -- the seeds are fixed, so the result is the same.

Needs: numpy
"""
import os, pickle
import numpy as np
import models as M
from common import OUTPUT

STEPS = 4000
LENS_TEST = list(range(2, 17))
STAGES = (("fixed", False), ("attention", True))


def _path(kind, name):
    return os.path.join(OUTPUT, "_%s_%s.pkl" % (kind, name))


def stage(name, att):
    cache = _path("cache", name)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("training the %s-context model" % name)
    rng = np.random.default_rng(1)
    m = M.Seq2Seq(d=48, attention=att, seed=2)
    hist = M.train(m, STEPS, rng, log_every=500, ckpt=_path("ckpt", name))
    acc = M.exact_match(m, np.random.default_rng(7), LENS_TEST)
    print("  exact match by length:", {k: round(v, 2) for k, v in acc.items()})
    out = dict(model=m, hist=hist, acc=acc)
    os.makedirs(OUTPUT, exist_ok=True)
    with open(cache + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(cache + ".tmp", cache)
    if os.path.exists(_path("ckpt", name)):
        os.remove(_path("ckpt", name))
    return out


def run():
    """Both models, each from its own cache stage (trained if absent)."""
    return {name: stage(name, att) for name, att in STAGES}
