"""The shared experiment of Class 28.

Ported from Class28_LSTM_Gated_RNNs/B_Recreated_Figures/src/common/
experiment.py, same seeds and settings: an RNN, an LSTM and a GRU of the
same width (d = 32) trained the same way on the recall task -- read 4 to
24 symbols, emit the first -- for 3,000 steps, tested to length 60, with
the gradient by lag read off the tape before and after training.

Cached stage by stage in output/_cache_<cell>.pkl (one stage per cell);
training also writes output/_ckpt_<cell>.pkl every 250 steps so an
interrupted run resumes exactly where it stopped.  Delete output/ to
recompute from scratch -- the seeds are fixed, so the result is the same.

Needs: numpy
"""
import os, pickle
import numpy as np
import models as M
from common import OUTPUT

STEPS = 3000
LENS_TEST = [4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 60]
CELLS = (M.RNNCell, M.LSTMCell, M.GRUCell)


def _path(kind, name):
    return os.path.join(OUTPUT, "_%s_%s.pkl" % (kind, name))


def stage(cell):
    cache = _path("cache", cell.kind)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("training the %s" % cell.kind)
    init = M.Recaller(cell, seed=3)
    lags0, g0 = M.grad_by_lag(init, np.random.default_rng(5))
    m = M.Recaller(cell, seed=3)
    hist = M.train(m, STEPS, np.random.default_rng(1), ckpt=_path("ckpt", cell.kind))
    acc = M.accuracy_by_length(m, np.random.default_rng(7), LENS_TEST)
    lags1, g1 = M.grad_by_lag(m, np.random.default_rng(5))
    print("  accuracy by length:", {k: round(v, 2) for k, v in acc.items()})
    out = dict(model=m, hist=hist, acc=acc, grad_init=(lags0, g0), grad_trained=(lags1, g1))
    os.makedirs(OUTPUT, exist_ok=True)
    with open(cache + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(cache + ".tmp", cache)
    if os.path.exists(_path("ckpt", cell.kind)):
        os.remove(_path("ckpt", cell.kind))
    return out


def run():
    """The three cells, each from its own cache stage (trained if absent)."""
    return {cell.kind: stage(cell) for cell in CELLS}
