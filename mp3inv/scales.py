"""Per-band lattice (scale) estimation and scalefactor decomposition.

An "entry" is one element of the decoder's sfb table (a long band, or one
window of a short band); each entry has its own scale 2**(q/4).
"""
import numpy as np

from . import lattice
from .lattice import PRETAB

Q_MAX = 41          # gg=255, k=0
Q_MIN = -170        # far finer than any sensible encoding


def entry_bounds(widths):
    c = np.concatenate([[0], np.cumsum(widths)])
    return c[:-1].astype(np.int64), c[1:].astype(np.int64)


def fit_entries(v, sigma, widths, T=6.0, q_range=(Q_MIN, Q_MAX)):
    """Coarsest-consistent-lattice fit for a batch of spectra sharing one sfb layout.

    v, sigma: [n, 576].  Returns dict of per-entry arrays [n, E]:
      q     coarsest q whose snap residuals are all <= T*sigma (Q_MAX+1 => zero entry)
      zmax  max normalised residual at q
      nnz   number of nonzero ix at q
      rho   scale / median sigma at q (regime indicator)
    """
    starts, ends = entry_bounds(widths)
    n, E = v.shape[0], len(widths)
    qbest = np.full((n, E), Q_MIN - 1, np.int64)
    done = np.zeros((n, E), bool)
    zbest = np.full((n, E), np.inf)
    nnz = np.zeros((n, E), np.int64)
    a = np.abs(v)
    # zero entries: consistent at the coarsest scale with everything snapped to 0
    z0 = a / sigma
    zmax0 = np.maximum.reduceat(z0, starts, axis=1)
    zero = zmax0 <= T
    qbest[zero] = Q_MAX + 1
    zbest[zero] = zmax0[zero]
    done |= zero
    for q in range(q_range[1], q_range[0] - 1, -1):
        if done.all():
            break
        s = np.float32(lattice.ideal_scale(q))
        ix, val = lattice.snap(v, s)
        z = np.abs(v - val) / sigma
        zm = np.maximum.reduceat(z, starts, axis=1)
        nz = np.add.reduceat((ix != 0).astype(np.int64), starts, axis=1)
        ok = (zm <= T) & ~done & (nz > 0)
        qbest[ok] = q
        zbest[ok] = zm[ok]
        nnz[ok] = nz[ok]
        done |= ok
    sig_e = np.stack([np.median(sigma[:, s:e], axis=1) for s, e in zip(starts, ends)], axis=1)
    rho = np.where(qbest <= Q_MAX, lattice.ideal_scale(np.minimum(qbest, Q_MAX)) / sig_e, np.inf)
    return dict(q=qbest, zmax=zbest, nnz=nnz, rho=rho)


LOG2E = 1.4426950408889634


def ix_bits(ix):
    """Crude Huffman cost model per coefficient (bits)."""
    a = np.abs(ix)
    return 2.0 * np.log2(1.0 + a) + (a > 0) + 0.25


def cost_curves(v, sigma, widths, q_lo=Q_MIN, q_hi=Q_MAX, kappa=1.0):
    """MDL cost (bits) of each entry at every q in [q_lo, q_hi].

    cost = log2(e) * sum r^2 / (2 (sigma/kappa)^2) + sum ix_bits(ix)
    Returns C [n, E, Q] with C[..., i] for q = q_lo + i.
    """
    starts, _ = entry_bounds(widths)
    n = v.shape[0]
    Q = q_hi - q_lo + 1
    C = np.zeros((n, len(widths), Q))
    w2 = (kappa / sigma) ** 2 * 0.5 * LOG2E
    for i, q in enumerate(range(q_lo, q_hi + 1)):
        s = np.float32(lattice.ideal_scale(q))
        ix, val = lattice.snap(v, s)
        c = (v - val) ** 2 * w2 + ix_bits(ix)
        C[:, :, i] = np.add.reduceat(c, starts, axis=1)
    return C


def decompose_curves(C, n_long, n_short, mixed, ms=0, q_lo=Q_MIN):
    """Choose side info minimising sum_j C[j, q_j] over all legal decompositions.

    C: [E, Q] cost curves of one granule/channel.  Returns dict (see decompose) with
    'q' per entry and 'cost'.
    """
    E, Q = C.shape
    maxsf, win, pre_t = entry_layout(n_long, n_short, mixed)
    q_hi = q_lo + Q - 1
    best = None
    sfr = np.arange(16)
    for sfsc in (0, 1):
        m = 2 << sfsc
        for preflag in ((0, 1) if not n_short else (0,)):
            pt = pre_t * preflag
            # candidate q for each (G, entry, sf): q = G - m*(sf+pt) - 8*sbg(window)
            Gs = np.arange(max(q_lo, -214 - 2 * ms), min(q_hi, 41 - 2 * ms) + 1)
            sbg_opts = range(8) if n_short else (0,)
            tot = np.zeros(len(Gs))
            chosen_sf = np.zeros((len(Gs), E), np.int64)
            chosen_sbg = np.zeros((len(Gs), 3), np.int64)
            groups = [(win == w) for w in range(3)] if n_short else []
            groups.append(win < 0)
            for gi, sel in enumerate(groups):
                if not sel.any():
                    continue
                idx = np.nonzero(sel)[0]
                bestg = None
                for sbg in (sbg_opts if gi < 3 and n_short else (0,)):
                    # q[G, j, sf]
                    qq = (Gs[:, None, None] - 8 * sbg - m * (sfr[None, None, :] + pt[idx][None, :, None]))
                    valid = sfr[None, None, :] <= maxsf[idx][None, :, None]
                    qi = np.clip(qq - q_lo, 0, Q - 1)
                    cc = C[idx][None, :, :][np.zeros_like(qi), np.arange(len(idx))[None, :, None], qi]
                    cc = np.where(valid & (qq >= q_lo) & (qq <= q_hi), cc, np.inf)
                    sfbest = np.argmin(cc, axis=2)
                    cbest = np.take_along_axis(cc, sfbest[:, :, None], axis=2)[:, :, 0].sum(axis=1)
                    if bestg is None:
                        bestg = [cbest, sfbest, np.full(len(Gs), sbg)]
                    else:
                        better = cbest < bestg[0]
                        bestg[0] = np.where(better, cbest, bestg[0])
                        bestg[1] = np.where(better[:, None], sfbest, bestg[1])
                        bestg[2] = np.where(better, sbg, bestg[2])
                tot += bestg[0]
                chosen_sf[:, idx] = bestg[1]
                if gi < 3 and n_short:
                    chosen_sbg[:, gi] = bestg[2]
            k = int(np.argmin(tot))
            cand = (tot[k], sfsc, preflag, int(Gs[k]))
            if best is None or cand < best[0]:
                G = int(Gs[k])
                sf = chosen_sf[k]
                sbg = chosen_sbg[k]
                q = G - m * (sf + pt) - np.where(win >= 0, 8 * sbg[np.maximum(win, 0)], 0)
                full = np.zeros(40, np.int64)
                full[:E] = sf
                best = (cand, dict(global_gain=G + 214 + 2 * ms, scalefac_scale=sfsc, preflag=preflag,
                                   subblock_gain=[int(v) for v in sbg], iscf=full, q=q, cost=float(tot[k])))
    return best[1]


def bits_proxy(v, q, widths):
    """Rough Huffman cost of snapping v at per-entry q (for hypothesis ranking)."""
    starts, ends = entry_bounds(widths)
    s = np.zeros_like(v)
    for j, (a, b) in enumerate(zip(starts, ends)):
        qq = np.minimum(q[:, j], Q_MAX)
        s[:, a:b] = lattice.ideal_scale(qq)[:, None]
    zero_entry = np.zeros_like(v, bool)
    for j, (a, b) in enumerate(zip(starts, ends)):
        zero_entry[:, a:b] = (q[:, j] > Q_MAX)[:, None]
    m = np.where(zero_entry, 0, np.abs(v) / s)
    ix = np.round(np.power(m, 0.75))
    return (2 * np.log2(1 + ix) + (ix > 0)).sum(axis=1)


# ------------------------------------------------------------- decomposition
def entry_layout(n_long, n_short, mixed):
    """Per entry: (max scalefactor, has_sf, window index or -1, pretab)."""
    E = n_long + n_short
    maxsf = np.zeros(E, np.int64)
    win = np.full(E, -1, np.int64)
    pre = np.zeros(E, np.int64)
    if not n_short:            # long: 0-10 slen1, 11-20 slen2, 21 none
        maxsf[:11] = 15
        maxsf[11:21] = 7
        pre[:] = PRETAB[:E]
    elif not n_long:           # short: 0-17 slen1, 18-35 slen2, 36-38 none
        maxsf[:18] = 15
        maxsf[18:36] = 7
        win[:] = np.arange(E) % 3
    else:                      # mixed: 8 long + 9 short slen1, 18 short slen2, 3 none
        maxsf[:17] = 15
        maxsf[17:35] = 7
        win[8:] = (np.arange(E - 8)) % 3
    return maxsf, win, pre


def decompose(qwant, n_long, n_short, mixed, ms=0, slack=None):
    """Find (global_gain, scalefac_scale, preflag, subblock_gain, iscf) realising
    per-entry exponents.

    qwant: [E] int, desired q per entry (> Q_MAX means zero entry: free).
    slack: [E] int >= 0, allowed extra fineness (q may be qwant - d, 0<=d<=slack),
           used for fine-regime entries.  Default 0 (exact).
    Returns dict or None.  Chooses the solution with minimal (total fineness, sf bits).
    """
    E = len(qwant)
    maxsf, win, pre_t = entry_layout(n_long, n_short, mixed)
    if slack is None:
        slack = np.zeros(E, np.int64)
    active = qwant <= Q_MAX
    best = None
    if not active.any():
        return dict(global_gain=210, scalefac_scale=0, preflag=0, subblock_gain=[0, 0, 0],
                    iscf=np.zeros(40, np.int64), q=np.full(E, 41 - 2 * ms), cost=0)
    qmax_act = int(qwant[active].max())
    has_sf = maxsf > 0
    for sfsc in (0, 1):
        m = 2 << sfsc
        for preflag in ((0, 1) if not n_short else (0,)):
            pt = pre_t * preflag
            for G in range(qmax_act, qmax_act + 8 * 7 + m * 18 + 1):
                gg = G + 214 + 2 * ms
                if gg > 255 or gg < 0:
                    continue
                # per window subblock gain choice (short entries); long entries: sbg = 0
                sbg = [0, 0, 0]
                iscf = np.zeros(E, np.int64)
                qs = np.full(E, 0, np.int64)
                total = 0
                feasible = True
                for w in (range(3) if n_short else ()):
                    sel = win == w
                    bw = None
                    for g_w in range(8):
                        res = _fit_group(G - 8 * g_w, qwant[sel], slack[sel], active[sel], maxsf[sel], has_sf[sel], m, pt[sel])
                        if res is not None and (bw is None or res[0] < bw[0]):
                            bw = (res[0], g_w, res[1], res[2])
                    if bw is None:
                        feasible = False
                        break
                    total += bw[0]
                    sbg[w] = bw[1]
                    iscf[sel] = bw[2]
                    qs[sel] = bw[3]
                if not feasible:
                    continue
                sel = win < 0
                if sel.any():
                    res = _fit_group(G, qwant[sel], slack[sel], active[sel], maxsf[sel], has_sf[sel], m, pt[sel])
                    if res is None:
                        continue
                    total += res[0]
                    iscf[sel] = res[1]
                    qs[sel] = res[2]
                cost = (total, sfsc, preflag, G)
                if best is None or cost < best[0]:
                    full = np.zeros(40, np.int64)
                    full[:E] = iscf
                    best = (cost, dict(global_gain=gg, scalefac_scale=sfsc, preflag=preflag,
                                       subblock_gain=list(sbg), iscf=full, q=qs, cost=total))
    return None if best is None else best[1]


def _fit_group(Gw, qwant, slack, active, maxsf, has_sf, m, pt):
    """Entries sharing base exponent Gw: q = Gw - m*(sf + pt).  Returns (cost, sf, q) or None."""
    n = len(qwant)
    sf = np.zeros(n, np.int64)
    q = Gw - m * pt
    cost = 0
    for i in range(n):
        if not active[i]:
            # zero entry: any sf; keep 0 (cheapest) unless pretab pushes q, irrelevant
            q[i] = Gw - m * pt[i]
            continue
        # need q = Gw - m*(sf+pt) in [qwant - slack, qwant]
        hi = qwant[i]
        lo = qwant[i] - slack[i]
        # sf + pt = (Gw - q)/m  -> smallest sf giving q <= hi
        need = Gw - hi                    # >= m*(sf+pt)
        if need < 0:
            return None
        t = -(-need // m)                 # ceil
        s = t - pt[i]
        if s < 0:
            s = 0
        if not has_sf[i] and s > 0:
            return None
        if s > maxsf[i]:
            return None
        qq = Gw - m * (s + pt[i])
        if qq < lo or qq > hi:
            return None
        sf[i] = s
        q[i] = qq
        cost += hi - qq
    return cost, sf, q
