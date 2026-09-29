"""Blind reconstruction pipeline: target PCM -> decoded parameters -> MP3 bytes.

Only the target WAV is used.  An optional `oracle` (dump of the original MP3)
is accepted purely for diagnostics and never influences decisions.
"""
import time

import numpy as np

from . import lattice, linmodel as LM, scales
from .bitstream import header_bytes, sfb_widths, write_mp3
from .linmodel import LONG, SHORT, STOP, MIXED
from .refdec import GR_DTYPE, Harness

SIGMA_E = np.sqrt(1.0 / 12.0)     # PCM rounding noise std (LSB)
# valid (class_g, class_g+1) pairs (ISO window sequencing; START is LONG followed by SHORT)
PAIRS = [(LONG, LONG), (LONG, SHORT), (SHORT, SHORT), (SHORT, STOP), (STOP, LONG)]
PAIRS_MIXED = [(LONG, MIXED), (MIXED, MIXED), (MIXED, STOP), (SHORT, MIXED), (MIXED, SHORT)]


def _bt_mx(c):
    return (2, 1) if c == MIXED else (int(c), 0)


def layout(sr, c):
    bt, mx = _bt_mx(c)
    return sfb_widths(sr, bt, mx)


class Analysis:
    """Holds subband estimates and the linear model for one channel."""

    def __init__(self, xch, sr):
        self.sr = sr
        self.A, self.B = LM.hybrid_blocks(sr)
        self.cpoly = np.sum(LM.poly_filters(sr) ** 2) / 32.0
        s = LM.poly_inverse(xch.astype(np.float64), iters=40)
        self.u = LM.granules_from_slots(s)          # [G, 576]
        self.G = self.u.shape[0]
        self._d = {}
        self._curves = {}

    def diag(self, c, c2):
        key = (c, c2)
        if key not in self._d:
            A, B = self.A[c], self.B[(c, c2)]
            self._d[key] = np.sum(A * A, axis=0) + np.sum(B * B, axis=0)
        return self._d[key]

    def spectra(self, c, c2, gs=None):
        """TDAC estimate of X_g assuming class c for g and c2 for g+1."""
        gs = np.arange(self.G) if gs is None else np.asarray(gs)
        u0 = self.u[gs]
        nxt = np.minimum(gs + 1, self.G - 1)
        u1 = np.where((gs + 1 < self.G)[:, None], self.u[nxt], 0.0)
        d = self.diag(c, c2)
        X = (u0 @ self.A[c] + u1 @ self.B[(c, c2)]) / d
        sigma = SIGMA_E / np.sqrt(d * self.cpoly)
        return X, np.broadcast_to(sigma, X.shape)


def choose_types(an, use_mixed=False, T=6.0):
    """Viterbi over window-sequence classes using MDL cost of the lattice fit."""
    pairs = PAIRS + (PAIRS_MIXED if use_mixed else [])
    classes = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    G = an.G
    cost = {}
    for (c, c2) in pairs:
        X, sig = an.spectra(c, c2)
        widths, _, _ = layout(an.sr, c)
        C = scales.cost_curves(X, sig, widths)
        cost[(c, c2)] = C.min(axis=2).sum(axis=1)
        an._curves[(c, c2)] = C
    INF = 1e18
    V = {c: np.full(G + 1, INF) for c in classes}
    arg = {c: np.zeros(G, np.int64) for c in classes}
    for c in classes:
        V[c][G] = 0.0
    for g in range(G - 1, -1, -1):
        for c in classes:
            best, bc = INF, -1
            for (a, b) in pairs:
                if a != c:
                    continue
                nxt = V[b][g + 1] if g + 1 < G else 0.0
                v = cost[(a, b)][g] + nxt
                if v < best:
                    best, bc = v, b
            V[c][g] = best
            arg[c][g] = bc
    c = min((LONG, SHORT), key=lambda k: V[k][0])
    seq = []
    for g in range(G):
        seq.append(c)
        c = arg[c][g] if g + 1 < G else c
    return np.array(seq)


def constrain_curves(C, v, sigma, widths, rho_hi=8.0, rho_lo=1.0, zero_T=4.0):
    """Regime handling: entries whose lattice is not identifiable are forced fine.

    identified : unconstrained MDL optimum has step/sigma >= rho_hi -> keep curve
    zero       : all |v| <= zero_T*sigma -> keep curve (optimum is all-zero)
    otherwise  : forbid q with step/sigma > rho_lo (fine-regime quantisation)
    """
    starts, ends = scales.entry_bounds(widths)
    C = C.copy()
    qgrid = np.arange(scales.Q_MIN, scales.Q_MAX + 1)
    regime = np.zeros(len(widths), np.int64)       # 0 zero, 1 identified, 2 forced fine
    for j, (a, b) in enumerate(zip(starts, ends)):
        sj = float(np.median(sigma[a:b]))
        if np.all(np.abs(v[a:b]) <= zero_T * sigma[a:b]):
            regime[j] = 0
            continue
        qh = qgrid[int(np.argmin(C[j]))]
        if lattice.ideal_scale(qh) / sj >= rho_hi:
            regime[j] = 1
            continue
        regime[j] = 2
        C[j, lattice.ideal_scale(qgrid) / sj > rho_lo] = np.inf
    return C, regime


def build_granules(an, types, T=6.0, ms=0, diag=None, rho_hi=8.0, rho_lo=1.0):
    """Estimate scales, decompose into side info and snap ix for every granule of one channel."""
    G = an.G
    recs = np.zeros(G, GR_DTYPE)
    info = []
    for g in range(G):
        c = types[g]
        c2 = types[g + 1] if g + 1 < G else LONG
        X, sig = an.spectra(c, c2, [g])
        widths, n_long, n_short = layout(an.sr, c)
        if (c, c2) in an._curves:
            C = an._curves[(c, c2)][g]
        else:
            C = scales.cost_curves(X, sig, widths)[0]
        if rho_hi is not None:
            C, regime = constrain_curves(C, X[0], sig[0], widths, rho_hi, rho_lo)
        else:
            regime = None
        dec = scales.decompose_curves(C, n_long, n_short, c == MIXED, ms)
        q = dec["q"]
        relaxed = False
        bt, mx = _bt_mx(c)
        if c == LONG and c2 in (SHORT, MIXED):
            bt = 1                           # START window for legality
        r = recs[g]
        r["block_type"], r["mixed_block_flag"] = bt, mx
        r["global_gain"] = dec["global_gain"]
        r["scalefac_scale"] = dec["scalefac_scale"]
        r["preflag"] = dec["preflag"]
        r["subblock_gain"] = dec["subblock_gain"]
        r["iscf"] = dec["iscf"]
        r["n_long_sfb"], r["n_short_sfb"] = n_long, n_short
        # exact per-entry scale floats and snapping
        S = lattice.scale_table()
        starts, ends = scales.entry_bounds(widths)
        maxsf, win, pre_t = scales.entry_layout(n_long, n_short, c == MIXED)
        ix = np.zeros(576, np.int64)
        shift = dec["scalefac_scale"] + 1
        for j, (a, b) in enumerate(zip(starts, ends)):
            isc = int(dec["iscf"][j]) + (pre_t[j] * dec["preflag"] if not n_short else 0)
            if win[j] >= 0:
                isc += dec["subblock_gain"][win[j]] << (3 - shift)
            k = isc << shift
            s = S[dec["global_gain"], ms, k] if k < 128 else np.float32(0)
            if s == 0:
                continue
            ixj, _ = lattice.snap(X[0, a:b], s)
            ix[a:b] = ixj
        r["ix"] = ix
        info.append(dict(q=q, relaxed=relaxed, cost=dec["cost"], regime=regime))
    return recs, info


def synth_check(recs_by_ch, sr, mode=3, mode_ext=0):
    """Parametric synthesis with the harness (s16).  recs_by_ch: [nch][G]."""
    H = Harness("s16")
    nch = len(recs_by_ch)
    G = len(recs_by_ch[0])
    hdr = header_bytes(sr, 9, 0, mode, mode_ext)
    st = H.new_state(hdr)
    grans = np.stack([np.asarray(r) for r in recs_by_ch], axis=1)   # [G, nch]
    hdrs = np.frombuffer(hdr * G, np.uint8).reshape(G, 4)
    return H.synth(st, hdrs, grans, nch)


def to_frames(recs_by_ch, mode=3, mode_ext=0):
    nch = len(recs_by_ch)
    G = len(recs_by_ch[0])
    frames = []
    for f in range(G // 2):
        frames.append((mode, mode_ext, [[recs_by_ch[ch][2 * f + gr] for ch in range(nch)] for gr in range(2)]))
    return frames


def reconstruct_mono(x, sr, log=print):
    t0 = time.time()
    an = Analysis(x[:, 0], sr)
    types = choose_types(an)
    log(f"types: {dict(zip(*np.unique(types, return_counts=True)))}  ({time.time() - t0:.1f}s)")
    recs, info = build_granules(an, types)
    log(f"granules built ({time.time() - t0:.1f}s); relaxed={sum(i['relaxed'] for i in info)}")
    y = synth_check([recs], sr)
    mism = np.nonzero(y[:, 0] != x[:, 0])[0]
    log(f"direct snap: {len(mism)} mismatching samples of {len(x)}")
    try:
        mp3 = write_mp3(to_frames([recs]), sr, 1)
        log(f"candidate size {len(mp3)} bytes = {len(mp3) * 8 / (len(x) / sr) / 1000:.0f} kbps")
    except Exception as e:  # noqa: BLE001
        log(f"writer: {e}")
    return recs, types, info, mism
