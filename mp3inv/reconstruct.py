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
        self.tail = 3
        self.tail_inflate = 16.0

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
        sigma = np.broadcast_to(sigma, X.shape).copy()
        # end of file: the synthesis tail is truncated, so the last granules are
        # poorly determined -- widen their noise so the fit prefers zeros there
        tail = gs >= self.G - self.tail
        sigma[tail] *= self.tail_inflate
        return X, sigma


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
            if g == G - 1 and c not in (LONG, STOP):
                continue          # a legal window sequence ends in a long or stop block
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


def refine_curves(C, v, sigma, widths, rho_refine=16.0, shift=16, min_nnz=4):
    """Superset refinement: for entries whose MDL lattice has step/sigma < rho_refine,
    only allow q <= q_hat - shift.  q_hat - 16 multiplies ix by 8 and (up to float
    rounding of the pow43 table) contains every point of the q_hat lattice, so the
    original values stay representable while many more near-solutions appear."""
    starts, ends = scales.entry_bounds(widths)
    C = C.copy()
    qgrid = np.arange(scales.Q_MIN, scales.Q_MAX + 1)
    refined = np.zeros(len(widths), bool)
    rho = np.full(len(widths), np.inf)
    for j, (a, b) in enumerate(zip(starts, ends)):
        k = int(np.argmin(C[j]))
        qh = qgrid[k]
        s = lattice.ideal_scale(qh)
        sj = float(np.median(sigma[a:b]))
        ix, _ = lattice.snap(v[a:b], np.float32(s))
        if np.count_nonzero(ix) < min_nnz:      # sparse band: its scale estimate is unreliable
            continue
        rho[j] = s / sj
        if rho[j] >= rho_refine or np.abs(ix).max() * 2 ** (3 * shift / 16) > 8100:
            continue
        C[j, qgrid > qh - shift] = np.inf
        refined[j] = True
    return C, refined, rho


def entry_scales(dec, widths, n_long, n_short, mixed, ms):
    """Exact decoder scale float for every coefficient, from a decomposition."""
    S = lattice.scale_table()
    starts, ends = scales.entry_bounds(widths)
    maxsf, win, pre_t = scales.entry_layout(n_long, n_short, mixed)
    out = np.zeros(576, np.float32)
    shift = dec["scalefac_scale"] + 1
    for j, (a, b) in enumerate(zip(starts, ends)):
        isc = int(dec["iscf"][j]) + (pre_t[j] * dec["preflag"] if not n_short else 0)
        if win[j] >= 0:
            isc += dec["subblock_gain"][win[j]] << (3 - shift)
        k = isc << shift
        out[a:b] = S[dec["global_gain"], ms, k] if k < 128 else np.float32(0)
    return out


class CodedSource:
    """Spectra of one *coded* channel (L, R, M or S per granule) with fixed block types,
    presented with the same interface as Analysis for build_granules."""

    def __init__(self, X, sigma, sr, tail):
        self.X, self.sigma, self.sr, self.tail = X, sigma, sr, tail
        self.G = X.shape[0]
        self._curves = {}

    def spectra(self, c, c2, gs):
        gs = np.asarray(gs)
        return self.X[gs], self.sigma[gs]


def build_granules(an, types, ms=0, rho_refine=8.0, shift=16, max_bits=3500, log=None):
    """Estimate scales, decompose into side info and snap ix for every granule of one coded
    channel.  `ms` is a scalar or a per-granule array (MS-stereo scale offset).

    Returns recs (GR_DTYPE [G]), per-granule info, and arrays xhat/sigma/scale_of [G, 576]."""
    from .bitstream import encode_granule
    G = an.G
    recs = np.zeros(G, GR_DTYPE)
    info = []
    xhat = np.zeros((G, 576))
    sigma = np.zeros((G, 576))
    scale_of = np.zeros((G, 576), np.float32)
    for g in range(G):
        msg = int(ms[g]) if np.ndim(ms) else int(ms)
        c = types[g]
        c2 = types[g + 1] if g + 1 < G else LONG
        X, sig = an.spectra(c, c2, [g])
        xhat[g], sigma[g] = X[0], sig[0]
        widths, n_long, n_short = layout(an.sr, c)
        if (c, c2) in an._curves:
            C0 = an._curves[(c, c2)][g]
        else:
            C0 = scales.cost_curves(X, sig, widths)[0]
        rr = rho_refine if g < G - an.tail - 1 else 0     # never refine the ill-conditioned tail
        while True:
            if rr and shift:
                C, refined, rho = refine_curves(C0, X[0], sig[0], widths, rr, shift)
            else:
                C, refined, rho = C0, np.zeros(len(widths), bool), None
            dec = scales.decompose_curves(C, n_long, n_short, c == MIXED, msg)
            if not np.isfinite(dec["cost"]) and rr:
                rr = rr / 2 if rr > 2 else 0
                continue
            sc = entry_scales(dec, widths, n_long, n_short, c == MIXED, msg)
            ix = np.where(sc > 0, lattice.snap(X[0], np.where(sc > 0, sc, 1))[0], 0)
            if rr and np.abs(ix).max() >= lattice.IX_MAX:     # clipped: refinement too fine here
                rr = rr / 2 if rr > 2 else 0
                continue
            tmp = np.zeros(1, GR_DTYPE)[0]
            bt, mx = _bt_mx(c)
            tmp["block_type"], tmp["mixed_block_flag"] = bt, mx
            tmp["iscf"] = dec["iscf"]
            tmp["ix"] = ix
            try:
                bits = encode_granule(tmp, an.sr)[1]["part_23_length"]
            except ValueError:
                bits = 1 << 20
            if bits <= max_bits or not rr:
                break
            rr = rr / 2 if rr > 2 else 0
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
        r["ix"] = ix
        scale_of[g] = sc
        info.append(dict(q=dec["q"], relaxed=rr != rho_refine, bits=bits, refined=int(refined.sum())))
    return recs, info, xhat, sigma, scale_of


# ------------------------------------------------------------------ state
class Candidate:
    """The evolving solution: decoded parameters for G granules x C coded channels."""

    def __init__(self, x, sr, recs, types, ms, xhat, sigma, scale_of):
        self.x = x                           # int16 [N, C]
        self.sr = sr
        self.recs = recs                     # GR_DTYPE [G, C]
        self.types = types                   # [C][G] block-type classes
        self.ms = np.asarray(ms, bool)       # [G]
        self.xhat, self.sigma, self.scale_of = xhat, sigma, scale_of   # [G, C, 576]
        self.G, self.C = recs.shape
        self.H = Harness("f32")
        self.hdrs = self._headers()

    def _headers(self):
        out = np.zeros((self.G, 4), np.uint8)
        for g in range(self.G):
            if self.C == 1:
                h = header_bytes(self.sr, 9, 0, 3, 0)
            else:
                h = header_bytes(self.sr, 9, 0, 1, 2 if self.ms[g] else 0)
            out[g] = np.frombuffer(h, np.uint8)
        return out

    def synth(self, ix=None):
        """Exact decoder float output [N, C] for coefficient array ix [G, C, 576]."""
        if ix is not None:
            self.recs["ix"] = ix
        st = self.H.new_state(bytes(self.hdrs[0]))
        return self.H.synth(st, self.hdrs, self.recs, self.C)

    def rescale(self, g, c):
        self.scale_of[g, c] = entry_scales(self.recs[g, c], *_lay(self.sr, self.types[c][g]),
                                           int(self.ms[g]))

    def frames(self):
        out = []
        for f in range(self.G // 2):
            g = 2 * f
            mode, ext = (3, 0) if self.C == 1 else (1, 2 if self.ms[g] else 0)
            out.append((mode, ext, [[self.recs[g + gr, ch] for ch in range(self.C)] for gr in range(2)]))
        return out


def _lay(sr, c):
    widths, n_long, n_short = layout(sr, c)
    return widths, n_long, n_short, c == MIXED


def mismatches(y, x, r0=0, r1=None):
    from .rounding import round_model
    r1 = len(x) if r1 is None else r1
    return int(np.count_nonzero(round_model(y[r0:r1]) != x[r0:r1]))


# ------------------------------------------------------------------- tail
def tail_resolve(cand, ntail=4, lam=1e-5, log=print, tail_inflated=3, inflate=16.0):
    """Re-estimate the last `ntail` granules by ridge least squares on the observed samples.

    The decoder output of the final granules extends past the end of the file, so the
    TDAC/polyphase inverse is ill-conditioned there.  Earlier granules are held at their
    current values; the tail coefficients of all coded channels jointly solve
        min ||M a - r||^2 + lam * mean||M_col||^2 ||a||^2
    over the samples that exist (both output channels; MS granules map a coded M value to
    (+,+) and S to (+,-)), which pulls unobservable directions to zero."""
    from .repair import responses
    G, C, N = cand.G, cand.C, len(cand.x)
    g0 = max(0, G - ntail)
    r0 = 576 * g0
    L = N - r0
    keep = cand.recs["ix"].copy()
    ix0 = keep.copy()
    ix0[g0:] = 0
    y0 = cand.synth(ix0)
    cand.recs["ix"] = keep
    r = np.concatenate([cand.x[r0:, oc].astype(np.float64) - y0[r0:, oc] for oc in range(C)])
    cols = []
    for g in range(g0, G):
        for c in range(C):
            R = responses(cand.types[c][g], cand.types[c][g + 1] if g + 1 < G else LONG, cand.sr)
            if C == 2 and cand.ms[g]:
                signs = (1.0, 1.0 if c == 0 else -1.0)
            else:
                signs = tuple(1.0 if oc == c else 0.0 for oc in range(C))
            st = 576 * g - r0
            n = min(L - st, R.shape[1])
            for i in range(576):
                col = np.zeros(C * L)
                for oc, sgn in enumerate(signs):
                    if sgn:
                        col[oc * L + st:oc * L + st + n] = sgn * R[i, :n]
                cols.append(col)
    M = np.stack(cols, axis=1)
    reg = lam * np.mean(np.sum(M * M, axis=0))
    a = np.linalg.solve(M.T @ M + reg * np.eye(M.shape[1]), M.T @ r).reshape(G - g0, C, 576)
    # re-estimate the tail scales from the ridge solution with the *uninflated* noise level
    # (the inflated one only served to keep the TDAC estimate from fitting garbage)
    sig = cand.sigma[g0:].copy()
    for k, g in enumerate(range(g0, G)):
        if g >= G - tail_inflated:
            sig[k] /= inflate
    for c in range(C):
        src = CodedSource(a[:, c], sig[:, c], cand.sr, tail=0)
        recs, _, xh, sg, sc = build_granules(src, list(cand.types[c][g0:]), ms=cand.ms[g0:].astype(int),
                                             rho_refine=0)
        # keep START legality w.r.t. the granule before the tail
        cand.recs[g0:, c] = recs
        cand.xhat[g0:, c], cand.sigma[g0:, c], cand.scale_of[g0:, c] = xh, sg, sc
    log(f"  tail: re-solved last {G - g0} granules by ridge LS")


# ----------------------------------------------------------- fallbacks
def exact_polish(cand, granules, log=print, max_evals=4000, back=2, deltas=(1, -1)):
    """Greedy coordinate search evaluated with the decoder itself (no model).

    For each failing PCM granule k, candidate moves are +-1 and zeroing on nonzero or
    near-threshold coefficients of granules k-2..k (all coded channels).  A move is kept iff
    the exact mismatch count on samples [576(k-2), end) decreases."""
    G, C = cand.G, cand.C
    evals = fixed = 0
    y = cand.synth()
    for k in granules:
        g0 = max(0, k - back)
        r0 = 576 * g0
        cur = mismatches(y, cand.x, r0)
        improved = True
        while cur and improved and evals < max_evals:
            improved = False
            moves = []
            for g in range(g0, min(G, k + 1)):
                for c in range(C):
                    ix = cand.recs["ix"][g, c]
                    s = cand.scale_of[g, c]
                    xh, sg = cand.xhat[g, c], cand.sigma[g, c]
                    idx = np.nonzero((s > 0) & ((ix != 0) | (np.abs(xh) >= 0.5 * s - 6 * sg)))[0]
                    for i in idx:
                        for nv in {int(ix[i]) + d for d in deltas} | {0}:
                            if nv != ix[i] and abs(nv) <= lattice.IX_MAX:
                                cost = abs(lattice.dequant(nv, s[i]) - xh[i]) / sg[i]
                                moves.append((cost, g, c, i, nv))
            moves.sort()
            for cost, g, c, i, nv in moves:
                if evals >= max_evals:
                    break
                old = cand.recs["ix"][g, c][i]
                cand.recs["ix"][g, c][i] = nv
                yt = cand.synth()
                evals += 1
                m = mismatches(yt, cand.x, r0)
                if m < cur:
                    cur, y = m, yt
                    fixed += 1
                    improved = True
                    if not cur:
                        break
                else:
                    cand.recs["ix"][g, c][i] = old
    log(f"  exact polish: {fixed} moves kept, {evals} decoder evaluations")
    return fixed


def scale_moves(cand, granules, log=print, max_nnz=4, passes=2, budget=3900):
    """Fallback for granules the integer repair cannot fix: the scale of a sparse band is
    ambiguous (a lone value fits several (ix, scale) pairs within noise).  Try every
    scalefactor value for sparse bands near the failure, re-snap that band, and keep
    changes that reduce the exact mismatch count."""
    fixed = 0
    y = cand.synth()
    recs = cand.recs
    for k in granules:
        g0 = max(0, k - 2)
        r0, r1 = 576 * g0, min(len(cand.x), 576 * (k + 1))
        cur = mismatches(y, cand.x, r0, r1)
        for _ in range(passes):
            if cur == 0:
                break
            best = None
            for g in range(g0, k + 1):
                for c in range(cand.C):
                    t = cand.types[c][g]
                    widths, n_long, n_short, mixed = _lay(cand.sr, t)
                    starts, ends = scales.entry_bounds(widths)
                    maxsf, _, _ = scales.entry_layout(n_long, n_short, mixed)
                    for j, (a, b) in enumerate(zip(starts, ends)):
                        nnz = int(np.count_nonzero(recs["ix"][g, c][a:b]))
                        loud = int(np.count_nonzero(np.abs(cand.xhat[g, c, a:b]) > 4 * cand.sigma[g, c, a:b]))
                        if maxsf[j] == 0 or max(nnz, loud) == 0 or max(nnz, loud) > max_nnz:
                            continue
                        old_sf = int(recs["iscf"][g, c][j])
                        old_ix = recs["ix"][g, c][a:b].copy()
                        for sf in range(maxsf[j] + 1):
                            if sf == old_sf:
                                continue
                            recs["iscf"][g, c][j] = sf
                            sc = entry_scales(recs[g, c], widths, n_long, n_short, mixed, int(cand.ms[g]))
                            if sc[a] == 0:
                                continue
                            recs["ix"][g, c][a:b] = lattice.snap(cand.xhat[g, c, a:b], sc[a])[0]
                            m = mismatches(cand.synth(), cand.x, r0, r1)
                            if m < cur and (best is None or m < best[0]):
                                best = (m, g, c, j, sf, recs["ix"][g, c][a:b].copy(), a, b)
                        recs["iscf"][g, c][j] = old_sf
                        recs["ix"][g, c][a:b] = old_ix
            if best is None:
                break
            m, g, c, j, sf, newix, a, b = best
            old_sf, old_ix = int(recs["iscf"][g, c][j]), recs["ix"][g, c][a:b].copy()
            recs["iscf"][g, c][j] = sf
            recs["ix"][g, c][a:b] = newix
            if granule_bits(recs[g, c], cand.sr) > budget:      # keep the file legal
                recs["iscf"][g, c][j] = old_sf
                recs["ix"][g, c][a:b] = old_ix
                break
            cand.rescale(g, c)
            y = cand.synth()
            cur = m
            fixed += 1
    log(f"  scale moves: {fixed} band scalefactors changed")
    return fixed


def granule_bits(rec, sr):
    from .bitstream import encode_granule
    try:
        return encode_granule(rec, sr)[1]["part_23_length"]
    except ValueError:
        return 1 << 20


def enforce_bit_limit(cand, limit=4095, log=print):
    """Legality guard: part2_3_length must fit in 12 bits.  If a granule is over the limit
    (rare; after refinement plus repair), drop its highest-frequency nonzero values until
    it fits.  This keeps the file legal at the price of exactness in that granule."""
    from .bitstream import encode_granule
    trimmed = 0
    for g in range(cand.G):
        for c in range(cand.C):
            r = cand.recs[g, c]
            while True:
                try:
                    bits = encode_granule(r, cand.sr)[1]["part_23_length"]
                except ValueError:
                    bits = limit + 1
                if bits <= limit:
                    break
                # prefer coarsening a high band's scalefactor (re-snapped from the estimate)
                # over zeroing values: the error stays at quantisation-noise level
                widths, n_long, n_short, mixed = _lay(cand.sr, cand.types[c][g])
                st, en = scales.entry_bounds(widths)
                cands_j = [j for j in range(len(widths)) if r["iscf"][j] > 0 and np.any(r["ix"][st[j]:en[j]])]
                if cands_j:
                    j = cands_j[-1]
                    r["iscf"][j] -= 1
                    cand.rescale(g, c)
                    sc = cand.scale_of[g, c]
                    r["ix"][st[j]:en[j]] = lattice.snap(cand.xhat[g, c, st[j]:en[j]], sc[st[j]])[0]
                else:
                    nz = np.nonzero(r["ix"])[0]
                    r["ix"][nz[-max(1, len(nz) // 50):]] = 0
                trimmed += 1
    if trimmed:
        log(f"  bit limit: trimmed {trimmed} times to stay within {limit} bits/granule")
    return trimmed


# ------------------------------------------------------------ top level
def choose_ms(anL, anR, typesL, typesR, sr, tail_frames=2):
    """Per frame: MS if both channels share block types and M/S fits the lattice more
    cheaply (MDL) than L/R.  Returns (ms [G] bool, X [G,2,576], sigma [G,2,576])."""
    G = anL.G
    XL = np.zeros((G, 576)); XR = np.zeros((G, 576)); SL = np.zeros((G, 576)); SR = np.zeros((G, 576))
    for g in range(G):
        for an, t, X, S in ((anL, typesL, XL, SL), (anR, typesR, XR, SR)):
            c2 = t[g + 1] if g + 1 < G else LONG
            Xg, Sg = an.spectra(t[g], c2, [g])
            X[g], S[g] = Xg[0], Sg[0]
    XM, XS = (XL + XR) / 2, (XL - XR) / 2
    SM = np.sqrt(SL ** 2 + SR ** 2) / 2
    cost = {}
    for name, X, S in (("L", XL, SL), ("R", XR, SR), ("M", XM, SM), ("S", XS, SM)):
        cst = np.zeros(G)
        tt = typesL if name != "R" else typesR
        for t in set(tt):
            sel = np.nonzero(np.asarray(tt) == t)[0]
            widths, _, _ = layout(sr, t)
            cst[sel] = scales.cost_curves(X[sel], S[sel], widths).min(axis=2).sum(axis=1)
        cost[name] = cst
    ms = np.zeros(G, bool)
    nf = G // 2
    for f in range(nf - tail_frames):
        g = slice(2 * f, 2 * f + 2)
        if np.any(np.asarray(typesL[g]) != np.asarray(typesR[g])):
            continue
        if (cost["M"][g] + cost["S"][g]).sum() < (cost["L"][g] + cost["R"][g]).sum():
            ms[g] = True
    # tail frames: the fit is meaningless there (inflated noise); inherit the last decision
    last = bool(ms[2 * (nf - tail_frames) - 1]) if nf > tail_frames else False
    for f in range(max(0, nf - tail_frames), nf):
        g = slice(2 * f, 2 * f + 2)
        ms[g] = last and np.all(np.asarray(typesL[g]) == np.asarray(typesR[g]))
    X = np.stack([np.where(ms[:, None], XM, XL), np.where(ms[:, None], XS, XR)], axis=1)
    S = np.stack([np.where(ms[:, None], SM, SL), np.where(ms[:, None], SM, SR)], axis=1)
    return ms, X, S


def reconstruct(x, sr, log=print, rho_refine=8.0, shift=16, repair=True, stereo_ms=True, repair_budget=300.0, keep=None):
    """Blind reconstruction from target PCM x (int16 [N, C]).  Returns (mp3 bytes, stats)."""
    from .repair import Repair
    N, C = x.shape
    if sr not in (32000, 44100, 48000):
        raise NotImplementedError("only MPEG-1 sample rates are implemented")
    if N % 1152:
        raise ValueError("target length must be a whole number of MPEG-1 frames (1152 samples)")
    if C > 2:
        raise ValueError("at most 2 channels")
    stats = {"channels": C, "samples": N, "sr": sr}
    t0 = time.time()
    ans = [Analysis(x[:, c], sr) for c in range(C)]
    types = [choose_types(an) for an in ans]
    G = ans[0].G
    stats["types"] = [{int(k): int(v) for k, v in zip(*np.unique(t, return_counts=True))} for t in types]
    log(f"block types {stats['types']}  ({time.time() - t0:.1f}s)")
    if C == 2 and stereo_ms:
        ms, X, S = choose_ms(ans[0], ans[1], types[0], types[1], sr)
        srcs = [CodedSource(X[:, c], S[:, c], sr, ans[c].tail) for c in range(2)]
        coded_types = [types[0], np.where(ms, types[0], types[1])]
    else:
        ms = np.zeros(G, bool)
        srcs = ans
        coded_types = types
    stats["ms_frames"] = int(ms[::2].sum())
    recs = np.zeros((G, C), GR_DTYPE)
    xhat = np.zeros((G, C, 576)); sigma = np.zeros((G, C, 576)); scale_of = np.zeros((G, C, 576), np.float32)
    refined = 0
    for c in range(C):
        r, info, xh, sg, sc = build_granules(srcs[c], coded_types[c], ms=ms.astype(int),
                                             rho_refine=rho_refine, shift=shift)
        recs[:, c], xhat[:, c], sigma[:, c], scale_of[:, c] = r, xh, sg, sc
        refined += sum(i["refined"] for i in info)
    stats["refined_entries"] = int(refined)
    cand = Candidate(x, sr, recs, coded_types, ms, xhat, sigma, scale_of)
    tail_resolve(cand, log=log)
    y = cand.synth()
    stats["mismatch_direct"] = mismatches(y, x)
    log(f"side info built ({time.time() - t0:.1f}s): {stats['ms_frames']} MS frames, {refined} bands refined; "
        f"direct snap -> {stats['mismatch_direct']} mismatching samples of {N * C}")
    if repair and stats["mismatch_direct"]:
        rp = Repair(cand, log=log)
        rp.time_budget = repair_budget
        rs = rp.run()
        for _ in range(3):
            if not rs["failures"]:
                break
            if not scale_moves(cand, sorted(set(rs["failures"])), log=log):
                break
            rs = rp.run()
        if rs["failures"]:
            exact_polish(cand, sorted(set(rs["failures"])), log=log)
        left = [k for k in range(G) if mismatches(cand.synth(), x, 576 * k, 576 * (k + 1))]
        if left and all(k >= G - 6 for k in left):
            # only the ill-conditioned tail is left: wider, deeper exact search there
            exact_polish(cand, left, log=log, max_evals=12000, back=3, deltas=(1, -1, 2, -2))
        stats["repair"] = {k: v for k, v in rs.items() if k != "failures"}
        stats["repair_failed_granules"] = rs["failures"][:50]
    stats["mismatch_final_model"] = mismatches(cand.synth(), x)
    stats["trimmed_granules"] = enforce_bit_limit(cand, log=log)
    if stats["trimmed_granules"]:
        stats["mismatch_final_model"] = mismatches(cand.synth(), x)
    if keep is not None:
        keep.append(cand)
    mp3 = write_mp3(cand.frames(), sr, C)
    stats["bytes"] = len(mp3)
    stats["kbps"] = round(len(mp3) * 8 / (N / sr) / 1000, 1)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"candidate: {len(mp3)} bytes ({stats['kbps']} kbps), "
        f"{stats['mismatch_final_model']} mismatches (harness), {stats['seconds']} s")
    return mp3, stats
