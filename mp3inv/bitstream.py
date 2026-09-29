"""Legal MPEG-1/2 Layer III bitstream writer.

Input is the *decoded parameter* representation (what actually reaches the
decoder's float pipeline): per granule/channel block type, mixed flag,
global_gain, scalefac_scale, preflag, subblock_gain, transmitted scalefactors
(iscf) and quantised values ix[576], plus per-frame stereo mode.  Everything
that does not affect decoded samples is chosen here to minimise bits:
Huffman tables, region split, big_values/count1 boundary, count1 table,
scalefac_compress, bit reservoir use and (optionally) per-frame bitrate.
scfsi is never used.
"""
from functools import lru_cache

import numpy as np

from . import huffman

BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
SR_TABLE = {44100: (1, 0), 48000: (1, 1), 32000: (1, 2),
            22050: (2, 0), 24000: (2, 1), 16000: (2, 2),
            11025: (25, 0), 12000: (25, 1), 8000: (25, 2)}
SLEN = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3)]
# MPEG-1 scalefactor partitions (entries of the sfb table), see minimp3 g_scf_partitions
PARTITIONS = {"long": (6, 5, 5, 5), "mixed": (8, 9, 6, 12), "short": (9, 9, 6, 12)}


class BitWriter:
    __slots__ = ("acc", "n")

    def __init__(self):
        self.acc, self.n = 0, 0

    def put(self, v, n):
        if n:
            self.acc = (self.acc << n) | (int(v) & ((1 << n) - 1))
            self.n += n

    def extend(self, other):
        self.acc = (self.acc << other.n) | other.acc
        self.n += other.n

    def tobytes(self, nbytes=None):
        nb = (self.n + 7) // 8 if nbytes is None else nbytes
        pad = nb * 8 - self.n
        assert pad >= 0
        return (self.acc << pad).to_bytes(nb, "big")


def header_bytes(sr, bitrate_idx, padding, mode, mode_ext, protection=1):
    ver, sri = SR_TABLE[sr]
    vid = {1: 3, 2: 2, 25: 0}[ver]
    w = BitWriter()
    w.put(0x7FF, 11); w.put(vid, 2); w.put(1, 2); w.put(protection, 1)
    w.put(bitrate_idx, 4); w.put(sri, 2); w.put(padding, 1); w.put(0, 1)
    w.put(mode, 2); w.put(mode_ext, 2); w.put(0, 1); w.put(1, 1); w.put(0, 2)
    return w.tobytes()


def frame_bytes(sr, kbps, padding):
    ver = SR_TABLE[sr][0]
    k = 144000 if ver == 1 else 72000
    return k * kbps // sr + padding


@lru_cache(None)
def sfb_widths(sr, block_type, mixed):
    """sfb table widths (decoder's own, via the harness) for this configuration."""
    from .refdec import Harness
    ver, sri = SR_TABLE[sr]
    hdr = header_bytes(sr, 9 if ver == 1 else 8, 0, 3, 0)
    w, nl, ns = Harness("s16").sfbtab(hdr, block_type, mixed)
    return tuple(int(v) for v in w), nl, ns


# ------------------------------------------------------------------ huffman
@lru_cache(None)
def _valid_tables():
    return [t for t in range(32) if t not in (4, 14)]


def _pair_prefix(ix, npairs):
    """prefix[t][p] = bits of pairs [0, p) under table t (1e9 per impossible pair)."""
    a = ix[0:2 * npairs:2]
    b = ix[1:2 * npairs:2]
    out = {}
    for t in _valid_tables():
        c = huffman.pair_bits(t, a, b)
        c = np.where(np.isinf(c), 1e9, c)
        out[t] = np.concatenate([[0.0], np.cumsum(c)])
    return out


def _best_table(prefix, p0, p1, maxv):
    """min-cost table for pairs [p0, p1) with max |value| maxv."""
    if p1 <= p0:
        return 0, 0.0
    best = (None, np.inf)
    for t in _valid_tables():
        if huffman.table_limit(t) < maxv:
            continue
        c = prefix[t][p1] - prefix[t][p0]
        if c < best[1]:
            best = (t, c)
    return best


def encode_huffman(ix, widths, window_switching, short_nomix):
    """Choose big_values/count1/regions/tables; return (BitWriter, side dict)."""
    ix = np.asarray(ix, np.int64)
    a = np.abs(ix)
    nz = np.nonzero(a)[0]
    last_nz = int(nz[-1]) + 1 if len(nz) else 0
    big = np.nonzero(a > 1)[0]
    last_big = int(big[-1]) + 1 if len(big) else 0
    start = last_big + (last_big & 1)
    cum = np.concatenate([[0], np.cumsum(widths)])
    maxabs_pref = np.maximum.accumulate(np.concatenate([[0], a])) if False else None
    best = None
    cands = sorted(set([start + 2 * k for k in range(0, 9)] + [last_nz + (last_nz & 1)]))
    full_prefix = _pair_prefix(ix, 288)
    for bv_end in cands:
        if bv_end > 576 or bv_end < start:
            continue
        n1 = max(0, last_nz - bv_end)
        c1_end = bv_end + 4 * ((n1 + 3) // 4)
        if c1_end > 576:
            continue
        # count1 cost
        if c1_end > bv_end:
            q = ix[bv_end:c1_end].reshape(-1, 4)
            c1costs = [int(huffman.count1_bits(t, q).sum()) for t in (0, 1)]
            c1t = int(np.argmin(c1costs))
            c1c = c1costs[c1t]
        else:
            c1t, c1c = 0, 0
        npairs = bv_end // 2

        def rmax(s, e):
            e = min(e, bv_end)
            return int(a[s:e].max()) if e > s else 0

        if window_switching:
            r0_end = int(cum[min(len(widths), 9 if short_nomix else 8)])
            e0 = min(r0_end, bv_end)
            t0, c0 = _best_table(full_prefix, 0, e0 // 2, rmax(0, e0))
            t1, c1 = _best_table(full_prefix, e0 // 2, npairs, rmax(e0, bv_end))
            reg = dict(table_select=[t0, t1, 0], region0_count=0, region1_count=0)
            bigc = c0 + c1
        else:
            bigc, reg = np.inf, None
            nb = len(widths)
            for r0 in range(16):
                e0 = int(cum[min(r0 + 1, nb)])
                if e0 >= bv_end and r0 > 0 and int(cum[min(r0, nb)]) >= bv_end:
                    break
                ce0 = min(e0, bv_end)
                t0, c0 = _best_table(full_prefix, 0, ce0 // 2, rmax(0, ce0))
                for r1 in range(8):
                    if r0 + r1 + 2 > nb:
                        break
                    e1 = int(cum[min(r0 + r1 + 2, nb)])
                    ce1 = min(e1, bv_end)
                    t1, c1 = _best_table(full_prefix, ce0 // 2, ce1 // 2, rmax(ce0, ce1))
                    t2, c2 = _best_table(full_prefix, ce1 // 2, npairs, rmax(ce1, bv_end))
                    c = c0 + c1 + c2
                    if c < bigc:
                        bigc, reg = c, dict(table_select=[t0, t1, t2], region0_count=r0, region1_count=r1)
                    if e1 >= bv_end:
                        break
        total = bigc + c1c
        if best is None or total < best[0]:
            best = (total, bv_end, c1_end, c1t, reg)
    total, bv_end, c1_end, c1t, reg = best
    assert total < 1e8, "value not encodable"
    # emit
    big_cb, c1_cb, lin = huffman.codebooks()
    w = BitWriter()
    ts = reg["table_select"]
    if window_switching:
        bounds = [int(cum[min(len(widths), 9 if short_nomix else 8)]), 576, 576]
    else:
        nb = len(widths)
        bounds = [int(cum[min(reg["region0_count"] + 1, nb)]),
                  int(cum[min(reg["region0_count"] + reg["region1_count"] + 2, nb)]), 576]
    for i in range(0, bv_end, 2):
        r = 0 if i < bounds[0] else (1 if i < bounds[1] else 2)
        t = ts[r]
        x, y = int(ix[i]), int(ix[i + 1])
        if t == 0:
            assert x == 0 and y == 0
            continue
        ax, ay = abs(x), abs(y)
        code, ln = big_cb[t][(min(ax, 15), min(ay, 15))]
        w.put(code, ln)
        for v, av in ((x, ax), (y, ay)):
            if lin[t] and av >= 15:
                w.put(av - 15, lin[t])
            if av:
                w.put(1 if v < 0 else 0, 1)
    for i in range(bv_end, c1_end, 4):
        q = [int(v) for v in ix[i:i + 4]]
        code, ln = c1_cb[c1t][tuple(1 if v else 0 for v in q)]
        w.put(code, ln)
        for v in q:
            if v:
                w.put(1 if v < 0 else 0, 1)
    side = dict(big_values=bv_end // 2, count1_table=c1t, **reg)
    return w, side


# -------------------------------------------------------------- scalefactors
def encode_scalefactors_mpeg1(iscf, n_long, n_short, mixed):
    kind = "long" if not n_short else ("mixed" if mixed else "short")
    parts = PARTITIONS[kind]
    n1 = parts[0] + parts[1]
    n2 = parts[2] + parts[3]
    v = [int(x) for x in iscf[:n1 + n2]]
    m1 = max(v[:n1]) if n1 else 0
    m2 = max(v[n1:n1 + n2]) if n2 else 0
    best = None
    for sc, (s1, s2) in enumerate(SLEN):
        if m1 > (1 << s1) - 1 or m2 > (1 << s2) - 1:
            continue
        bits = n1 * s1 + n2 * s2
        if best is None or bits < best[0]:
            best = (bits, sc)
    if best is None:
        raise ValueError(f"scalefactors not representable: max {m1}/{m2}")
    sc = best[1]
    s1, s2 = SLEN[sc]
    w = BitWriter()
    for i, x in enumerate(v):
        w.put(x, s1 if i < n1 else s2)
    return w, sc


# ------------------------------------------------------------------- frames
def encode_granule(g, sr, mpeg1=True):
    bt, mixed = int(g["block_type"]), int(g["mixed_block_flag"])
    widths, n_long, n_short = sfb_widths(sr, bt, mixed)
    if not mpeg1:
        raise NotImplementedError("LSF writer not implemented")
    sw, sc = encode_scalefactors_mpeg1(g["iscf"], n_long, n_short, mixed)
    hw, side = encode_huffman(g["ix"], widths, bt != 0, bt == 2 and not mixed)
    part23 = sw.n + hw.n
    if part23 >= 4096:
        raise ValueError(f"granule needs {part23} bits (> 4095)")
    sw.extend(hw)
    side.update(part_23_length=part23, scalefac_compress=sc)
    return sw, side


def side_info_bits(side, nch, mpeg1, main_data_begin):
    w = BitWriter()
    ngr = 2 if mpeg1 else 1
    if mpeg1:
        w.put(main_data_begin, 9)
        w.put(0, 5 if nch == 1 else 3)
        w.put(0, 4 * nch)   # scfsi
    else:
        w.put(main_data_begin, 8)
        w.put(0, 1 if nch == 1 else 2)
    for gr in range(ngr):
        for ch in range(nch):
            s, g = side[gr][ch]
            w.put(s["part_23_length"], 12)
            w.put(s["big_values"], 9)
            w.put(int(g["global_gain"]), 8)
            w.put(s["scalefac_compress"], 4 if mpeg1 else 9)
            bt = int(g["block_type"])
            if bt:
                w.put(1, 1); w.put(bt, 2); w.put(int(g["mixed_block_flag"]), 1)
                w.put(s["table_select"][0], 5); w.put(s["table_select"][1], 5)
                for k in range(3):
                    w.put(int(g["subblock_gain"][k]), 3)
            else:
                w.put(0, 1)
                for k in range(3):
                    w.put(s["table_select"][k], 5)
                w.put(s["region0_count"], 4); w.put(s["region1_count"], 3)
            if mpeg1:
                w.put(int(g["preflag"]), 1)
            w.put(int(g["scalefac_scale"]), 1)
            w.put(s["count1_table"], 1)
    return w


def write_mp3(frames, sr, nch, bitrate="vbr", max_kbps=320):
    """frames: list of (mode, mode_ext, grans) with grans indexable [gr][ch] of GR_DTYPE records.

    bitrate: "vbr" (smallest legal per-frame bitrate using the reservoir) or an int kbps (CBR).
    Returns the MP3 byte string.
    """
    ver = SR_TABLE[sr][0]
    mpeg1 = ver == 1
    ngr = 2 if mpeg1 else 1
    rates = BITRATES[1 if mpeg1 else 2]
    maxres = 511 if mpeg1 else 255
    si_len = (17 if nch == 1 else 32) if mpeg1 else (9 if nch == 1 else 17)
    enc = []
    for (mode, mode_ext, grans) in frames:
        bits, side = BitWriter(), [[None] * nch for _ in range(ngr)]
        for gr in range(ngr):
            for ch in range(nch):
                w, s = encode_granule(grans[gr][ch], sr, mpeg1)
                side[gr][ch] = (s, grans[gr][ch])
                bits.extend(w)
        enc.append((mode, mode_ext, side, bits))
    M = [(b.n + 7) // 8 for (_, _, _, b) in enc]
    allowed = [i for i, r in enumerate(rates) if r and r <= max_kbps]
    if bitrate != "vbr":
        allowed = [rates.index(int(bitrate))]
    cap = {i: frame_bytes(sr, rates[i], 0) - 4 - si_len for i in allowed}
    cmax = max(cap.values())
    n = len(enc)
    req = [0] * (n + 1)
    for f in range(n - 1, -1, -1):
        req[f] = max(0, M[f] + req[f + 1] - cmax)
        if req[f] > maxres:
            raise ValueError(f"frame {f}: needs {req[f]} reservoir bytes (> {maxres}); bit budget exceeded")
    out = bytearray()
    R = 0
    for f, (mode, mode_ext, side, bits) in enumerate(enc):
        choice = None
        for i in allowed:
            if R + cap[i] - M[f] >= req[f + 1] and M[f] <= R + cap[i]:
                choice = i
                break
        assert choice is not None
        C = cap[choice]
        mdb = R
        hdr = header_bytes(sr, choice, 0, mode, mode_ext)
        si = side_info_bits(side, nch, mpeg1, mdb).tobytes(si_len)
        # main data of this frame: first `mdb` bytes live in previous frames' slots
        data = bits.tobytes(M[f])
        # we emit the stream as slots: previous frame slots were padded; place data
        out_frame_slot = bytearray(C)
        enc[f] = (hdr, si, data, mdb, C)
        R = min(maxres, R + C - M[f])
    # second pass: lay out main data stream across slots
    stream_len = sum(e[4] for e in enc)
    stream = bytearray(stream_len)
    S = 0
    for (hdr, si, data, mdb, C) in enc:
        D = S - mdb
        stream[D:D + len(data)] = data
        S += C
    S = 0
    for (hdr, si, data, mdb, C) in enc:
        out += hdr + si + stream[S:S + C]
        S += C
    return bytes(out)
