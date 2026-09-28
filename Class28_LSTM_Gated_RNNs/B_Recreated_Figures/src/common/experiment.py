"""The shared experiment: an RNN, an LSTM and a GRU of the same width
(d = 32) trained the same way on the recall task -- read 4 to 24
symbols, emit the first -- and tested to length 60.  Cached in
../figs/_cache.pkl."""
import os, pickle
import numpy as np
from . import models as M

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figs", "_cache.pkl")
STEPS = 3000
LENS_TEST = [4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 60]
CELLS = (M.RNNCell, M.LSTMCell, M.GRUCell)


def run(force=False):
    if os.path.exists(CACHE) and not force:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    out = {}
    for cell in CELLS:
        print("training the %s" % cell.kind)
        init = M.Recaller(cell, seed=3)
        lags0, g0 = M.grad_by_lag(init, np.random.default_rng(5))
        m = M.Recaller(cell, seed=3)
        hist = M.train(m, STEPS, np.random.default_rng(1))
        acc = M.accuracy_by_length(m, np.random.default_rng(7), LENS_TEST)
        lags1, g1 = M.grad_by_lag(m, np.random.default_rng(5))
        print("  accuracy by length:", {k: round(v, 2) for k, v in acc.items()})
        out[cell.kind] = dict(model=m, hist=hist, acc=acc, grad_init=(lags0, g0), grad_trained=(lags1, g1))
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out
