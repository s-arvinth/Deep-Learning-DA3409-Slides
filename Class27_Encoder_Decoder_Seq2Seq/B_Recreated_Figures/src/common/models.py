"""The objects Class 27 computes with.

`Seq2Seq`        an RNN encoder-decoder in the notation of Jurafsky &
                 Martin (2026) Section 14.7, with the context fed to
                 every decoder step (Eq. 14.32) and, optionally,
                 dot-product attention (Eqs. 14.35-14.37)
`reverse_batch`  the task every experiment uses: read a sequence of
                 digits, write it reversed -- so the output length equals
                 the input length but the alignment is not local, which
                 is what the encoder-decoder is for
`train`          teacher-forced training with Adam and gradient clipping
`greedy_decode`, `beam_decode`
                 the two decoders of Jurafsky Section 13.4
`exact_match`    accuracy of the whole output sequence, by length

Everything is trained here, on the tape in `common/tape.py`; nothing is
loaded from a library.
"""
import numpy as np
from . import tape as tp

PAD, BOS, EOS = 0, 10, 11
V = 12                      # 0 pad, 1-9 digits, 10 <s>, 11 </s>


def reverse_batch(rng, n, B):
    """B source sequences of length n over the digits 1-9, and their
    targets: the reversal, followed by </s>."""
    src = rng.integers(1, 10, size=(B, n))
    tgt = np.concatenate([src[:, ::-1], np.full((B, 1), EOS)], axis=1)
    return src, tgt


class Seq2Seq:
    def __init__(self, d=48, attention=False, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        self.d = d; self.attention = attention
        self.E = tp.T(rng.normal(size=(V, d)) * 0.3)
        self.We = tp.T(rng.normal(size=(d, d)) * s); self.Ue = tp.T(_orth(rng, d))
        self.be = tp.T(np.zeros(d))
        self.Wd = tp.T(rng.normal(size=(2 * d, d)) * s); self.Ud = tp.T(_orth(rng, d))
        self.bd = tp.T(np.zeros(d))
        self.Vo = tp.T(rng.normal(size=(d, V)) * s); self.bo = tp.T(np.zeros(V))
        self.params = [self.E, self.We, self.Ue, self.be, self.Wd, self.Ud, self.bd,
                       self.Vo, self.bo]

    # ---- the encoder: h_t = tanh(U h_{t-1} + W e_t + b), Jurafsky Eq. 14.5
    def encode(self, src):
        B, n = src.shape
        h = tp.T(np.zeros((B, self.d)))
        H = []
        for t in range(n):
            e = self.E[src[:, t]]
            h = tp.tanh(h @ self.Ue + e @ self.We + self.be)
            H.append(h)
        return H

    # ---- the context: Jurafsky Eq. 14.33 (fixed) or Eqs. 14.35-14.37
    def context(self, H, h_prev):
        if not self.attention:
            return H[-1], None
        scores = tp.stack([(h_prev * Hj).sum(axis=1) for Hj in H], axis=1)   # (B, n)
        alpha = tp.softmax(scores, axis=1)
        c = None
        for j, Hj in enumerate(H):
            term = alpha[:, j:j + 1] * Hj
            c = term if c is None else c + term
        return c, alpha

    # ---- one decoder step: h^d_t = g(y_{t-1}, h^d_{t-1}, c), Jurafsky Eq. 14.32
    def step(self, H, h, y_prev):
        c, alpha = self.context(H, h)
        e = self.E[y_prev]
        h = tp.tanh(h @ self.Ud + tp.concat([e, c], axis=1) @ self.Wd + self.bd)
        logits = h @ self.Vo + self.bo
        return h, logits, alpha

    def loss(self, src, tgt):
        """Teacher-forced cross-entropy per target token (Jurafsky
        Section 14.7.1): the decoder reads the gold y_{t-1}, not its own
        output."""
        H = self.encode(src)
        B, m = tgt.shape
        h = H[-1]
        y_prev = np.full(B, BOS)
        losses = []
        for t in range(m):
            h, logits, _ = self.step(H, h, y_prev)
            losses.append(tp.cross_entropy(logits, tgt[:, t]))
            y_prev = tgt[:, t]
        total = losses[0]
        for l in losses[1:]:
            total = total + l
        return total * (1.0 / m)

    # ---- inference
    def greedy_decode(self, src, max_len=None):
        H = self.encode(src)
        B, n = src.shape
        max_len = max_len or n + 1
        h = H[-1]; y_prev = np.full(B, BOS)
        out = []; alphas = []
        for t in range(max_len):
            h, logits, alpha = self.step(H, h, y_prev)
            y_prev = np.argmax(logits.v, axis=1)
            out.append(y_prev)
            if alpha is not None:
                alphas.append(alpha.v)
        return np.stack(out, axis=1), (np.stack(alphas, axis=1) if alphas else None)

    def beam_decode(self, src1, k=2, max_len=None):
        """Beam search over one source (Jurafsky Section 13.4): keep the
        k most probable prefixes, extend each, keep the k best again.
        Returns (best sequence, its log probability, the search tree)."""
        H = self.encode(src1[None, :])
        n = src1.shape[0]; max_len = max_len or n + 1
        beams = [([], 0.0, H[-1], BOS, False)]
        tree = []
        for t in range(max_len):
            cand = []
            for seq, lp, h, y_prev, done in beams:
                if done:
                    cand.append((seq, lp, h, y_prev, True)); continue
                h2, logits, _ = self.step(H, h, np.array([y_prev]))
                logp = logits.v[0] - np.log(np.exp(logits.v[0] - logits.v[0].max()).sum()) - logits.v[0].max()
                top = np.argsort(-logp)[:k + 1]
                for y in top:
                    cand.append((seq + [int(y)], lp + float(logp[y]), h2, int(y), y == EOS))
                tree.append((t, seq, [(int(y), float(logp[y])) for y in top]))
            cand.sort(key=lambda c: -c[1])
            beams = cand[:k]
            if all(b[4] for b in beams):
                break
        best = beams[0]
        return best[0], best[1], tree

    def sequence_logprob(self, src1, seq):
        """log p(seq | src) under the model, for scoring a decoded output."""
        H = self.encode(src1[None, :]); h = H[-1]; y_prev = BOS; lp = 0.0
        for y in seq:
            h, logits, _ = self.step(H, h, np.array([y_prev]))
            z = logits.v[0]; lp += float(z[y] - z.max() - np.log(np.exp(z - z.max()).sum()))
            y_prev = y
        return lp


def _orth(rng, d):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def train(model, steps, rng, lens=(2, 10), B=32, lr=3e-3, clip=5.0, log_every=250):
    opt = tp.Adam(model.params, lr=lr, clip=clip)
    hist = []
    for s in range(1, steps + 1):
        n = int(rng.integers(lens[0], lens[1] + 1))
        src, tgt = reverse_batch(rng, n, B)
        L = model.loss(src, tgt)
        opt.zero(); L.backward(); opt.step()
        hist.append(float(L.v))
        if log_every and s % log_every == 0:
            print("  step %5d  loss %.3f" % (s, np.mean(hist[-log_every:])))
    return hist


def exact_match(model, rng, lengths, B=200):
    """Fraction of sources whose greedy decoding equals the target, for
    each length in `lengths`."""
    out = {}
    for n in lengths:
        src, tgt = reverse_batch(rng, n, B)
        pred, _ = model.greedy_decode(src)
        out[n] = float(np.mean(np.all(pred == tgt, axis=1)))
    return out
