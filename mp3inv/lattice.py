"""The decoder's exact dequantisation lattice.

value(ix, scale) = sign(ix) * float32(pow43_dec(|ix|) * scale), where pow43_dec is
minimp3's L3_pow_43 (table below 129, polynomial approximation above) and scale is
F(global_gain, ms, k) computed by chained L3_ldexp_q2 calls.
"""
from functools import lru_cache

import numpy as np

from .refdec import Harness

IX_MAX = 8191 + 15
PRETAB = np.array([0] * 11 + [1, 1, 1, 1, 2, 2, 3, 3, 3, 2] + [0], np.int64)


@lru_cache(None)
def pow43_table():
    H = Harness("s16")
    return np.array([H.pow43(i) for i in range(IX_MAX + 2)], np.float32)


@lru_cache(None)
def scale_table():
    """S[gg, ms, k] float32 for gg 0..255, ms 0..1, k 0..127."""
    H = Harness("s16")
    S = np.zeros((256, 2, 128), np.float32)
    for gg in range(256):
        for ms in range(2):
            for k in range(128):
                S[gg, ms, k] = H.scale(gg, ms, k)
    return S


def q_of(gg, ms, k):
    """Quarter-step exponent: scale ~= 2**(q/4)."""
    return gg - 214 - 2 * ms - k


def ideal_scale(q):
    return 2.0 ** (np.asarray(q, np.float64) / 4.0)


def snap(v, scale):
    """Nearest decoder lattice point for values v (float64 array) at float32 scale(s).

    Returns (ix signed int64, lattice value float64)."""
    tab = pow43_table().astype(np.float64)
    s = np.asarray(scale, np.float64)
    a = np.abs(v)
    m = a / s
    n0 = np.floor(np.power(np.maximum(m, 0), 0.75)).astype(np.int64)
    n0 = np.clip(n0, 0, IX_MAX - 1)
    best_n, best_e = None, None
    for d in (-1, 0, 1, 2):
        n = np.clip(n0 + d, 0, IX_MAX)
        val = (tab[n] * s).astype(np.float32).astype(np.float64)
        e = np.abs(a - val)
        if best_n is None:
            best_n, best_e = n, e
        else:
            better = e < best_e
            best_n = np.where(better, n, best_n)
            best_e = np.where(better, e, best_e)
    ix = np.where(v < 0, -best_n, best_n)
    val = (tab[best_n] * s).astype(np.float32).astype(np.float64)
    return ix, np.where(v < 0, -val, val)


def dequant(ix, scale):
    tab = pow43_table()
    a = np.abs(np.asarray(ix))
    val = (tab[a] * np.asarray(scale, np.float32)).astype(np.float32).astype(np.float64)
    return np.where(np.asarray(ix) < 0, -val, val)
