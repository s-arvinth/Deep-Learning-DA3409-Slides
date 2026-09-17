"""The objects Class 24 computes with.

Nothing in this class is statistical, so nothing here is random: every
figure is an exact computation, and every number on a slide comes from
these functions.

`iou`, `nms`                  the box metric and the suppression rule
                              of Bishop & Bishop (2024), Sections 10.4.2
                              and 10.4.5
`window_cost`                 multiply-adds for the sliding-window
                              network of Bishop Figs. 10.22-10.23, done
                              naively and done once as a convolution
`conv2d_matrix`,
`transpose_conv`              a strided 2-D convolution as a matrix, and
                              the up-sampling that is its transpose
`avgpool`, `maxpool`,
`unpool_avg`, `unpool_max`    the four resolution changes of Bishop
                              Figs. 10.28-10.29
`vgg19_spec`, `layer_params`  the published VGG-19 configuration and
                              its exact parameter count
`hourglass_spec`              the encoder-decoder schedule of Noh et al.
                              (2015) as Prince (2023), Section 10.5.3
                              describes it
`synthetic_image`             the constructed scene of Class 23, reused
"""
import numpy as np


# ------------------------------------------------------------ boxes
def iou(a, b):
    """Intersection over union of two boxes given as (x, y, w, h) with
    (x, y) the centre, in any units."""
    ax0, ax1 = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay0, ay1 = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx0, bx1 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by0, by1 = b[1] - b[3] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, p_min=0.7, iou_max=0.5):
    """Non-max suppression as Bishop Section 10.4.5 states it, for one
    class: discard boxes below p_min, then repeatedly keep the most
    probable remaining box and discard every box whose IoU with it
    exceeds iou_max.  Returns the indices kept, in the order kept."""
    alive = [i for i in range(len(boxes)) if scores[i] >= p_min]
    kept = []
    while alive:
        best = max(alive, key=lambda i: scores[i])
        kept.append(best)
        alive = [i for i in alive
                 if i != best and iou(boxes[i], boxes[best]) <= iou_max]
    return kept


# ------------------------------------------- the sliding-window network
def window_cost(n_img, n_win=6, k=3, pool=2):
    """Multiply-adds to run Bishop's toy network (k x k convolution,
    non-overlapping pool x pool pooling, one fully connected output) on
    the n_win x n_win windows of an n_img x n_img image.

    Because the pooling is non-overlapping, the enlarged network of
    Fig. 10.23 evaluates windows at a stride of `pool`, so the naive
    count uses the same window positions.

    naive:   one full forward pass per window position
    shared:  one enlarged network over the whole image, whose last layer
             is the fully connected layer applied as a small convolution

    Returns (positions, naive, shared).  Pooling costs no multiply-adds.
    """
    c_win = n_win - k + 1                     # conv map side, one window
    p_win = c_win // pool                     # pooled side, one window
    per_window = c_win ** 2 * k ** 2 + p_win ** 2
    positions = ((n_img - n_win) // pool + 1) ** 2
    c_img = n_img - k + 1                     # conv map side, whole image
    p_img = c_img // pool
    fc_side = p_img - p_win + 1               # the FC layer as a p_win-kernel conv
    assert fc_side ** 2 == positions
    shared = c_img ** 2 * k ** 2 + fc_side ** 2 * p_win ** 2
    return positions, positions * per_window, shared


# --------------------------------------------- transpose convolution
def conv2d_matrix(n, W, s):
    """The n x n -> m x m strided convolution (no padding) as a matrix
    of shape (m*m, n*n)."""
    k = W.shape[0]
    m = (n - k) // s + 1
    M = np.zeros((m * m, n * n))
    for oi in range(m):
        for oj in range(m):
            for a in range(k):
                for b in range(k):
                    M[oi * m + oj, (oi * s + a) * n + (oj * s + b)] = W[a, b]
    return M, m


def transpose_conv(Z, W, s, n):
    """Up-sampling by scattering: every input unit multiplies the kernel
    into a k x k patch of the output, patches stepping by s, and
    overlaps are summed.  Returns the n x n output."""
    k = W.shape[0]
    out = np.zeros((n, n))
    m = Z.shape[0]
    for i in range(m):
        for j in range(m):
            out[i * s:i * s + k, j * s:j * s + k] += Z[i, j] * W
    return out


# --------------------------------------------------------- resolution
def avgpool(X, p=2):
    n = X.shape[0] // p
    return X[:n * p, :n * p].reshape(n, p, n, p).mean(axis=(1, 3))


def maxpool(X, p=2):
    n = X.shape[0] // p
    B = X[:n * p, :n * p].reshape(n, p, n, p)
    out = B.max(axis=(1, 3))
    where = np.zeros_like(X, bool)
    for i in range(n):
        for j in range(n):
            blk = X[i * p:(i + 1) * p, j * p:(j + 1) * p]
            a, b = np.unravel_index(np.argmax(blk), blk.shape)
            where[i * p + a, j * p + b] = True
    return out, where


def unpool_avg(Z, p=2):
    return np.kron(Z, np.ones((p, p)))


def unpool_max(Z, p=2, where=None):
    """Max-unpooling: the value goes to the first cell of its block, or
    to the remembered position if `where` is given."""
    n = Z.shape[0]
    out = np.zeros((n * p, n * p))
    for i in range(n):
        for j in range(n):
            if where is None:
                out[i * p, j * p] = Z[i, j]
            else:
                blk = where[i * p:(i + 1) * p, j * p:(j + 1) * p]
                a, b = np.argwhere(blk)[0]
                out[i * p + a, j * p + b] = Z[i, j]
    return out


def bilinear_up(X, f=2):
    """Bilinear up-sampling by a factor f, edges clamped."""
    n = X.shape[0]
    g = (np.arange(n * f) + 0.5) / f - 0.5
    g = np.clip(g, 0, n - 1)
    i0 = np.floor(g).astype(int); i1 = np.minimum(i0 + 1, n - 1); t = g - i0
    R = X[i0, :] * (1 - t)[:, None] + X[i1, :] * t[:, None]
    C = R[:, i0] * (1 - t)[None, :] + R[:, i1] * t[None, :]
    return C


# -------------------------------------------- published architectures
# VGG-19, Simonyan & Zisserman (2015), configuration E: (kind, out
# channels).  Kernel 3 x 3 everywhere; pooling halves the side.
VGG19 = ([("conv", 64)] * 2 + [("pool", None)]
         + [("conv", 128)] * 2 + [("pool", None)]
         + [("conv", 256)] * 4 + [("pool", None)]
         + [("conv", 512)] * 4 + [("pool", None)]
         + [("conv", 512)] * 4 + [("pool", None)]
         + [("fc", 4096), ("fc", 4096), ("fc", 1000)])


def vgg19_spec(side=224, cin=3):
    """Walk the configuration: returns a list of dicts with the layer
    kind, spatial side, channels and exact parameter count."""
    out = []
    c, s = cin, side
    for kind, ch in VGG19:
        if kind == "conv":
            params = 3 * 3 * c * ch + ch
            c = ch
        elif kind == "pool":
            s = s // 2
            params = 0
        else:
            n_in = s * s * c if not out or out[-1]["kind"] != "fc" else c
            params = n_in * ch + ch
            c, s = ch, 1
        out.append(dict(kind=kind, side=s, channels=c, params=params))
    return out


def hourglass_spec():
    """The encoder-decoder of Noh et al. (2015) as Prince Section 10.5.3
    gives it: a VGG encoder to 7 x 7, two fully connected layers of
    4096, a fully connected layer back to 7 x 7 x 512, then a mirror of
    the encoder up to 224 x 224 x 21.  Returns (labels, sides, chans)."""
    enc = [(224, 3), (224, 64), (112, 128), (56, 256), (28, 512), (14, 512),
           (7, 512)]
    mid = [(1, 4096), (1, 4096)]
    dec = [(7, 512), (14, 512), (28, 512), (56, 256), (112, 128), (224, 64),
           (224, 21)]
    seq = enc + mid + dec
    return [s for s, _ in seq], [c for _, c in seq], len(enc), len(mid)


# ------------------------------------------------- the scene of Class 23
def synthetic_image(n=64):
    """A constructed scene: a bright disc, a dark square, a soft ramp and
    a thin line, on a mid-grey ground."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.full((n, n), 0.45)
    img[(x - 0.30) ** 2 + (y - 0.35) ** 2 < 0.045] = 0.90
    img[(np.abs(x - 0.72) < 0.14) & (np.abs(y - 0.68) < 0.14)] = 0.10
    img += 0.25 * np.clip((x - 0.55) / 0.45, 0, 1) * (y < 0.30)
    img[np.abs(y - 0.85) < 0.012] = 0.95
    return np.clip(img, 0, 1)


# ------------------------------------------------- checked on import
_Z = np.array([[1.0, 2.0], [3.0, 4.0]])
assert np.allclose(avgpool(unpool_avg(_Z)), _Z)
assert np.allclose(maxpool(unpool_max(_Z))[0], _Z)
