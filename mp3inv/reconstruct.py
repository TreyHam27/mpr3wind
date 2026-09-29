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


def build_granules(an, types, ms=0, rho_refine=8.0, shift=16, max_bits=4000, log=None):
    """Estimate scales, decompose into side info and snap ix for every granule of one channel.

    Returns recs (GR_DTYPE [G]), per-granule info, and arrays xhat/sigma/scale_of [G, 576]."""
    from .bitstream import encode_granule
    G = an.G
    recs = np.zeros(G, GR_DTYPE)
    info = []
    xhat = np.zeros((G, 576))
    sigma = np.zeros((G, 576))
    scale_of = np.zeros((G, 576), np.float32)
    for g in range(G):
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
            dec = scales.decompose_curves(C, n_long, n_short, c == MIXED, ms)
            if not np.isfinite(dec["cost"]) and rr:
                rr = rr / 2 if rr > 2 else 0
                continue
            sc = entry_scales(dec, widths, n_long, n_short, c == MIXED, ms)
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
        q = dec["q"]
        relaxed = rr != rho_refine
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
        info.append(dict(q=q, relaxed=relaxed, bits=bits, refined=int(refined.sum()), rho=rho))
    return recs, info, xhat, sigma, scale_of


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


def reconstruct(x, sr, log=print, **kw):
    """Blind reconstruction from target PCM x (int16 [N, ch]).  Returns (mp3 bytes, stats).

    Stereo is handled as independent L/R channels (header mode 'stereo'); each channel
    decodes exactly as it would in a mono stream (verified by the tests), so the mono
    machinery is reused per channel.
    """
    N, nch = x.shape
    if sr not in (32000, 44100, 48000):
        raise NotImplementedError("only MPEG-1 sample rates are implemented")
    if N % 1152:
        raise ValueError("target length must be a whole number of MPEG-1 frames (1152 samples)")
    stats = {"channels": nch, "samples": N, "sr": sr}
    if nch == 1:
        mp3, recs, st = reconstruct_mono(x, sr, log=log, **kw)
        stats.update(st)
        return mp3, stats
    all_recs = []
    stats["per_channel"] = []
    for ch in range(nch):
        log(f"--- channel {ch}")
        _, recs, st = reconstruct_mono(x[:, ch:ch + 1], sr, log=log, **kw)
        all_recs.append(recs)
        stats["per_channel"].append(st)
    mp3 = write_mp3(to_frames(all_recs, mode=0), sr, nch)
    stats["bytes"] = len(mp3)
    stats["kbps"] = round(len(mp3) * 8 / (N / sr) / 1000, 1)
    return mp3, stats


def float_synth_fn(recs, sr, mode=3, mode_ext=0):
    """Returns f(ix [G,576]) -> exact decoder pre-rounding output [N] for one channel."""
    H = Harness("f32")
    G = len(recs)
    hdr = header_bytes(sr, 9, 0, mode, mode_ext)
    hdrs = np.frombuffer(hdr * G, np.uint8).reshape(G, 4)
    base = np.array(recs, copy=True)

    def f(ix):
        base["ix"] = ix
        st = H.new_state(hdr)
        return H.synth(st, hdrs, base[:, None], 1)[:, 0]
    f.base = base
    return f


def tail_resolve(x, recs, fs, xhat, scale_of, types, sr, ntail=4, lam=1e-5, log=print):
    """Re-estimate the last `ntail` granules by ridge least squares on the observed samples.

    The decoder's output for the final granules extends past the end of the file, so the
    TDAC/polyphase inverse is ill-conditioned there.  Here the earlier granules are held
    at their current values and the tail coefficients solve
        min ||M a - r||^2 + lam * ||M||^2 ||a||^2
    over the samples that exist, which pulls unobservable directions to zero."""
    from .repair import responses
    G = len(recs)
    g0 = max(0, G - ntail)
    N = len(x)
    r0 = 576 * g0
    ix0 = recs["ix"].copy()
    ix0[g0:] = 0
    y0 = fs(ix0)
    r = x[r0:].astype(np.float64) - y0[r0:]
    cols = []
    for g in range(g0, G):
        R = responses(types[g], types[g + 1] if g + 1 < G else LONG, sr)
        for i in range(576):
            col = np.zeros(N - r0)
            st = 576 * g - r0
            L = min(len(col) - st, R.shape[1])
            col[st:st + L] = R[i, :L]
            cols.append(col)
    M = np.stack(cols, axis=1)
    reg = lam * np.mean(np.sum(M * M, axis=0))
    a = np.linalg.solve(M.T @ M + reg * np.eye(M.shape[1]), M.T @ r)
    a = a.reshape(G - g0, 576)
    for k, g in enumerate(range(g0, G)):
        sc = scale_of[g]
        xhat[g] = a[k]
        recs["ix"][g] = np.where(sc > 0, lattice.snap(a[k], np.where(sc > 0, sc, 1))[0], 0)
    log(f"  tail: re-solved last {G - g0} granules by ridge LS")


def exact_polish(x, recs, fs, xhat, sigma, scale_of, granules, log=print, max_evals=4000, zmax=6.0):
    """Greedy coordinate search evaluated with the decoder itself (no model).

    For each failing PCM granule k, candidate moves are +-1 and zeroing on coefficients of
    granules k-2..k that are nonzero or within zmax*sigma of a nonzero lattice value.
    A move is kept iff the exact mismatch count on samples [576(k-2), end) decreases."""
    from .rounding import round_model
    G = len(recs)
    N = len(x)
    evals = 0
    fixed = 0
    y = fs(recs["ix"])
    for k in granules:
        g0 = max(0, k - 2)
        r0 = 576 * g0
        cur = int(np.count_nonzero(round_model(y[r0:]) != x[r0:]))
        improved = True
        while cur and improved and evals < max_evals:
            improved = False
            moves = []
            for g in range(g0, min(G, k + 1)):
                ix = recs["ix"][g]
                s = scale_of[g]
                near = (np.abs(xhat[g]) > (np.abs(xhat[g]) - zmax * sigma[g]).clip(0)) & (s > 0)
                idx = np.nonzero((ix != 0) | (np.abs(xhat[g]) >= s * 0.5 - zmax * sigma[g]) & near)[0]
                for i in idx:
                    for nv in {int(ix[i]) + 1, int(ix[i]) - 1, 0}:
                        if nv != ix[i] and abs(nv) <= lattice.IX_MAX:
                            cost = abs(lattice.dequant(nv, s[i]) - xhat[g, i]) / sigma[g, i]
                            moves.append((cost, g, i, nv))
            moves.sort()
            for cost, g, i, nv in moves:
                if evals >= max_evals:
                    break
                old = recs["ix"][g][i]
                recs["ix"][g][i] = nv
                yt = fs(recs["ix"])
                evals += 1
                m = int(np.count_nonzero(round_model(yt[r0:]) != x[r0:]))
                if m < cur:
                    cur, y = m, yt
                    fixed += 1
                    improved = True
                    if not cur:
                        break
                else:
                    recs["ix"][g][i] = old
    log(f"  exact polish: {fixed} moves kept, {evals} decoder evaluations")
    return fixed


def rec_scales(rec, sr, types_c, ms=0):
    widths, n_long, n_short = layout(sr, types_c)
    return entry_scales(rec, widths, n_long, n_short, types_c == MIXED, ms)


def scale_moves(x, recs, fs, xhat, sigma, types, sr, granules, log=print, max_nnz=4, passes=2):
    """Fallback for granules the integer repair cannot fix: the scale of a sparse band is
    ambiguous (a lone value fits several (ix, scale) pairs within noise).  Try every
    scalefactor value for sparse bands near the failure, re-snap that band, and keep
    changes that reduce the exact mismatch count."""
    from .rounding import round_model
    base = fs.base
    G = len(recs)
    fixed = 0
    y = fs(recs["ix"])
    for k in granules:
        g0 = max(0, k - 2)
        r0, r1 = 576 * g0, min(len(x), 576 * (k + 1))
        cur = int(np.count_nonzero(round_model(y[r0:r1]) != x[r0:r1]))
        for _ in range(passes):
            if cur == 0:
                break
            best = None
            for g in range(g0, k + 1):
                c = types[g]
                widths, n_long, n_short = layout(sr, c)
                starts, ends = scales.entry_bounds(widths)
                maxsf, _, _ = scales.entry_layout(n_long, n_short, c == MIXED)
                for j, (a, b) in enumerate(zip(starts, ends)):
                    nnz = int(np.count_nonzero(recs["ix"][g][a:b]))
                    loud = int(np.count_nonzero(np.abs(xhat[g, a:b]) > 4 * sigma[g, a:b]))
                    if maxsf[j] == 0 or max(nnz, loud) == 0 or max(nnz, loud) > max_nnz:
                        continue
                    old_sf = int(recs["iscf"][g][j])
                    old_ix = recs["ix"][g][a:b].copy()
                    for sf in range(maxsf[j] + 1):
                        if sf == old_sf:
                            continue
                        recs["iscf"][g][j] = sf
                        sc = rec_scales(recs[g], sr, c)
                        if sc[a] == 0:
                            continue
                        recs["ix"][g][a:b] = lattice.snap(xhat[g, a:b], sc[a])[0]
                        base["iscf"][g] = recs["iscf"][g]
                        yt = fs(recs["ix"])
                        m = int(np.count_nonzero(round_model(yt[r0:r1]) != x[r0:r1]))
                        if m < cur and (best is None or m < best[0]):
                            best = (m, g, j, sf, recs["ix"][g][a:b].copy())
                    recs["iscf"][g][j] = old_sf
                    base["iscf"][g] = recs["iscf"][g]
                    recs["ix"][g][a:b] = old_ix
            if best is None:
                break
            m, g, j, sf, newix = best
            a, b = scales.entry_bounds(layout(sr, types[g])[0])[0][j], scales.entry_bounds(layout(sr, types[g])[0])[1][j]
            recs["iscf"][g][j] = sf
            base["iscf"][g] = recs["iscf"][g]
            recs["ix"][g][a:b] = newix
            y = fs(recs["ix"])
            cur = m
            fixed += 1
    log(f"  scale moves: {fixed} band scalefactors changed")
    return fixed


def reconstruct_mono(x, sr, log=print, rho_refine=8.0, shift=16, repair=True, stats=None):
    """Blind reconstruction of a mono target.  Returns (mp3 bytes, recs, stats)."""
    from .repair import MonoRepair
    from .rounding import round_model
    stats = {} if stats is None else stats
    t0 = time.time()
    an = Analysis(x[:, 0], sr)
    types = choose_types(an)
    stats["types"] = {int(k): int(v) for k, v in zip(*np.unique(types, return_counts=True))}
    log(f"block types {stats['types']}  ({time.time() - t0:.1f}s)")
    recs, info, xhat, sigma, scale_of = build_granules(an, types, rho_refine=rho_refine, shift=shift)
    stats["refined_entries"] = int(sum(i["refined"] for i in info))
    stats["refine_reduced_granules"] = int(sum(i["relaxed"] for i in info))
    fs = float_synth_fn(recs, sr)
    tail_resolve(x[:, 0], recs, fs, xhat, scale_of, types, sr, log=log)
    y = fs(recs["ix"])
    stats["mismatch_direct"] = int(np.count_nonzero(round_model(y) != x[:, 0]))
    log(f"scales/side info built ({time.time() - t0:.1f}s): {stats['refined_entries']} bands refined; "
        f"direct snap -> {stats['mismatch_direct']} mismatching samples of {len(x)}")
    if repair and stats["mismatch_direct"]:
        rp = MonoRepair(x[:, 0], types, xhat, sigma, scale_of, fs, sr, log=log)
        ix, rs = rp.run(recs["ix"].astype(np.int64).copy())
        recs["ix"] = ix
        for rnd in range(3):
            if not rs["failures"]:
                break
            fails = sorted(set(rs["failures"]))
            if not scale_moves(x[:, 0], recs, fs, xhat, sigma, types, sr, fails, log=log):
                break
            for g in range(len(recs)):
                scale_of[g] = rec_scales(recs[g], sr, types[g])
            ix, rs = rp.run(recs["ix"].astype(np.int64).copy())
            recs["ix"] = ix
        if rs["failures"]:
            exact_polish(x[:, 0], recs, fs, xhat, sigma, scale_of, sorted(set(rs["failures"])), log=log)
        stats["repair"] = {k: v for k, v in rs.items() if k != "failures"}
        stats["repair_failed_granules"] = rs["failures"][:50]
    y = fs(recs["ix"])
    stats["mismatch_final_model"] = int(np.count_nonzero(round_model(y) != x[:, 0]))
    mp3 = write_mp3(to_frames([recs]), sr, 1)
    stats["bytes"] = len(mp3)
    stats["kbps"] = round(len(mp3) * 8 / (len(x) / sr) / 1000, 1)
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"candidate: {len(mp3)} bytes ({stats['kbps']} kbps), "
        f"{stats['mismatch_final_model']} mismatches (harness), {stats['seconds']} s")
    return mp3, recs, stats
