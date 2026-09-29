"""Huffman codebooks derived directly from minimp3's decoding tables.

The encode tables are obtained by enumerating the decoder's own lookup trees
(parsed from minimp3.h), so they are consistent with the reference decoder by
construction.
"""
import re
from functools import lru_cache

import numpy as np

from . import MINIMP3_H


def _parse_array(src, name):
    m = re.search(r"static const (?:int16_t|uint8_t) " + name + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise ValueError(name)
    return [int(v) for v in re.findall(r"-?\d+", m.group(1))]


@lru_cache(None)
def raw_tables():
    src = open(MINIMP3_H).read()
    body = src[src.index("static void L3_huffman("):]
    return {k: _parse_array(body, k) for k in ("tabs", "tab32", "tab33", "tabindex", "g_linbits")}


def _enum_big(tabs, base):
    """Enumerate {(x, y): (code, length)} of a big-value table rooted at tabs[base]."""
    codes = {}

    def walk(off, w, prefix, plen):
        for idx in range(1 << w):
            leaf = tabs[off + idx]
            if leaf >= 0:
                n = leaf >> 8          # bits consumed at this level
                if idx & ((1 << (w - n)) - 1):
                    continue           # duplicate entry for a shorter code
                code = (prefix << n) | (idx >> (w - n))
                key = (leaf & 15, (leaf >> 4) & 15)
                if key not in codes:
                    codes[key] = (code, plen + n)
            else:
                walk(base - (leaf >> 3), leaf & 7, (prefix << w) | idx, plen + w)

    walk(base, 5, 0, 0)
    return codes


def _enum_count1(tab):
    codes = {}
    for idx in range(16):
        leaf = tab[idx]
        if leaf & 8:
            n = leaf & 7
            if idx & ((1 << (4 - n)) - 1):
                continue
            code, ln = idx >> (4 - n), n
            key = tuple((leaf >> (7 - s)) & 1 for s in range(4))
            codes.setdefault(key, (code, ln))
        else:
            w = leaf & 3
            for j in range(1 << w):
                l2 = tab[(leaf >> 3) + j]
                n = l2 & 7     # total length from start
                extra = n - 4
                if j & ((1 << (w - extra)) - 1):
                    continue
                code = (idx << extra) | (j >> (w - extra))
                key = tuple((l2 >> (7 - s)) & 1 for s in range(4))
                codes.setdefault(key, (code, n))
    return codes


# maximum |value| codable without linbits, per table (0 = no code)
TABLE_MAX = [0, 1, 2, 2, 0, 3, 3, 5, 5, 5, 7, 7, 7, 15, 0, 15] + [15] * 16


@lru_cache(None)
def codebooks():
    """Return (big, count1, linbits): big[t] = dict or None, count1[0|1] = dict."""
    t = raw_tables()
    big = []
    for tn in range(32):
        if tn in (0, 4, 14):
            big.append(None)
            continue
        big.append(_enum_big(t["tabs"], t["tabindex"][tn]))
    return big, [_enum_count1(t["tab32"]), _enum_count1(t["tab33"])], t["g_linbits"]


@lru_cache(None)
def cost_tables():
    """cost[t] : int array [16, 16] of code lengths (without sign/linbits); -1 = impossible."""
    big, _, _ = codebooks()
    out = []
    for tn in range(32):
        c = np.full((16, 16), -1, np.int64)
        if big[tn] is not None:
            for (x, y), (_, ln) in big[tn].items():
                c[x, y] = ln
        out.append(c)
    return out


def table_limit(tn):
    """Largest |value| encodable with table tn (inf-safe int)."""
    _, _, lin = codebooks()
    if tn == 0:
        return 0
    if tn < 16:
        return TABLE_MAX[tn]
    return 15 + (1 << lin[tn]) - 1


def pair_bits(tn, a, b):
    """Vectorised bit cost of pairs (a, b) (signed ints) under table tn; inf where impossible."""
    a = np.abs(np.asarray(a))
    b = np.abs(np.asarray(b))
    if tn == 0:
        return np.where((a == 0) & (b == 0), 0, np.inf)
    lim = table_limit(tn)
    _, _, lin = codebooks()
    ok = (a <= lim) & (b <= lim)
    ac = np.minimum(a, 15)
    bc = np.minimum(b, 15)
    c = cost_tables()[tn][ac, bc].astype(np.float64)
    c = np.where(c < 0, np.inf, c)
    bits = c + (a > 0) + (b > 0)
    if lin[tn]:
        bits = bits + lin[tn] * ((a >= 15).astype(np.int64) + (b >= 15))
    return np.where(ok, bits, np.inf)


def count1_bits(table, q):
    """q: int array [n, 4] of values in {-1,0,1}; returns bits per quad."""
    _, c1, _ = codebooks()
    lens = np.zeros(16, np.int64)
    for key, (_, ln) in c1[table].items():
        lens[key[0] * 8 + key[1] * 4 + key[2] * 2 + key[3]] = ln
    a = np.abs(q)
    idx = a[:, 0] * 8 + a[:, 1] * 4 + a[:, 2] * 2 + a[:, 3]
    return lens[idx] + a.sum(axis=1)
