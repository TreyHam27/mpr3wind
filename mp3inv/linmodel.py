"""float64 linear model of minimp3's synthesis, identified by impulse injection.

Two stages, matching the decoder:
  hybrid:    spectra X_g (576, bitstream order, after stereo) -> subband samples
             u_g (32 x 18, grbuf layout sb*18+slot).  u_g = A[t_g] X_g + B[t_{g-1}, t_g] X_{g-1}
  polyphase: subband samples (slot-major) -> PCM.  y[32t+i] = sum_j P_j[i,k] s_k[t-j]

The operators are measured from the harness (the decoder's own code), so the
model reproduces the decoder's quirks (e.g. start-window handling) exactly, up
to float32 rounding inside the decoder.
"""
from functools import lru_cache

import numpy as np

from .bitstream import header_bytes
from .refdec import Harness

# block-type "classes" as seen by the decoder's float pipeline
LONG, START, SHORT, STOP, MIXED = 0, 1, 2, 3, 4
TYPES = (LONG, START, SHORT, STOP, MIXED)
AMP = 1024.0


def _bt(t):
    return (2, 1) if t == MIXED else (t, 0)


@lru_cache(None)
def poly_filters(sr=44100):
    """P[j] (16, 32, 32): output sample i of slot t from subband k of slot t-j."""
    F = Harness("f32")
    hdr = header_bytes(sr, 9 if sr >= 32000 else 8, 0, 3, 0)
    P = np.zeros((17, 32, 32))
    for k in range(32):
        st = F.new_state(hdr)
        sb = np.zeros((3, 1, 576), np.float32)
        sb[1, 0, k * 18] = AMP
        y = F.poly(st, sb, 1)[:, 0] / AMP / 32768.0 * 32768.0
        r = y[576:576 + 17 * 32]
        P[:, :, k] = r.reshape(17, 32)
    assert np.abs(P[16]).max() == 0
    return P[:16]


@lru_cache(None)
def hybrid_blocks(sr=44100):
    """A[t] (576x576) and B[(tp, t)] (576x576): subband output (flattened sb*18+slot)."""
    F = Harness("f32")
    hdr = header_bytes(sr, 9 if sr >= 32000 else 8, 0, 3, 0)
    hdrs = np.frombuffer(hdr * 2, np.uint8).reshape(2, 4)
    A, B = {}, {}
    for t in TYPES:
        bt0, mx0 = _bt(t)
        for t2 in TYPES:
            bt1, mx1 = _bt(t2)
            Mc = np.zeros((576, 576))
            Mn = np.zeros((576, 576))
            for k in range(576):
                st = F.new_state(hdr)
                xr = np.zeros((2, 1, 576), np.float32)
                xr[0, 0, k] = AMP
                _, sb = F.synth_xr(st, hdrs, xr, [[bt0], [bt1]], [[mx0], [mx1]], want_sb=True)
                Mc[:, k] = sb[0, 0].reshape(576) / AMP
                Mn[:, k] = sb[1, 0].reshape(576) / AMP
            A[t] = Mc
            B[(t, t2)] = Mn
    return A, B


# ----------------------------------------------------------------- polyphase
def poly_synth(s, P=None):
    """s: [T, 32] subband samples (global slot index) -> y [T*32]."""
    P = poly_filters() if P is None else P
    T = s.shape[0]
    y = np.zeros((T, 32))
    for j in range(16):
        if j < T:
            y[j:] += s[:T - j] @ P[j].T
    return y.reshape(-1)


def poly_adjoint(y, P=None):
    P = poly_filters() if P is None else P
    T = len(y) // 32
    Y = y.reshape(T, 32)
    s = np.zeros((T, 32))
    for j in range(16):
        if j < T:
            s[:T - j] += Y[j:] @ P[j]
    return s


def cgls(op, adj, b, x0, iters=30, tol=1e-12):
    """Least squares min ||op(x) - b|| by CGLS."""
    x = x0.copy()
    r = b - op(x)
    s = adj(r)
    p = s.copy()
    gamma = np.vdot(s, s)
    g0 = gamma
    for _ in range(iters):
        q = op(p)
        alpha = gamma / max(np.vdot(q, q), 1e-300)
        x += alpha * p
        r -= alpha * q
        s = adj(r)
        gnew = np.vdot(s, s)
        if gnew <= tol * g0:
            break
        p = s + (gnew / gamma) * p
        gamma = gnew
    return x


def poly_inverse(y, iters=30, P=None):
    P = poly_filters() if P is None else P
    c = np.sum(P ** 2) / 32.0      # ~ ||column||^2
    x0 = poly_adjoint(y, P) / c
    return cgls(lambda s: poly_synth(s, P), lambda r: poly_adjoint(r, P), y, x0, iters)


# -------------------------------------------------------------------- hybrid
def slots_from_granules(u):
    """u: [G, 576] grbuf layout (sb*18+slot) -> s [G*18, 32]."""
    G = u.shape[0]
    return u.reshape(G, 32, 18).transpose(0, 2, 1).reshape(G * 18, 32)


def granules_from_slots(s):
    G = s.shape[0] // 18
    return s.reshape(G, 18, 32).transpose(0, 2, 1).reshape(G, 576)


def hybrid_synth(X, types, A=None, B=None):
    """X: [G, 576] spectra, types: [G] block-type classes -> u [G, 576]."""
    if A is None:
        A, B = hybrid_blocks()
    G = X.shape[0]
    u = np.zeros((G, 576))
    for g in range(G):
        u[g] = A[types[g]] @ X[g]
        if g:
            u[g] += B[(types[g - 1], types[g])] @ X[g - 1]
    return u


def hybrid_adjoint(u, types, A=None, B=None):
    if A is None:
        A, B = hybrid_blocks()
    G = u.shape[0]
    X = np.zeros((G, 576))
    for g in range(G):
        X[g] = A[types[g]].T @ u[g]
        if g + 1 < G:
            X[g] += B[(types[g], types[g + 1])].T @ u[g + 1]
    return X


def synth(X, types):
    """Full model: spectra [G,576] -> PCM [G*576] (one channel, already stereo-processed)."""
    return poly_synth(slots_from_granules(hybrid_synth(X, types)))


def adjoint(y, types):
    return hybrid_adjoint(granules_from_slots(poly_adjoint(y)), types)
