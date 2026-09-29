"""Exact-PCM repair of a candidate parameter set.

Given decoded parameters whose synthesis mismatches the target on some samples,
search alternative lattice values for low-confidence coefficients so that every
sample lands in its rounding interval.  The search is a MILP (scipy/HiGHS):

  y_n(z) = y_n(current, exact decoder float) + sum_{i,k} z_ik * dv_ik * h_i(n)
  lo_n <= y_n(z) <= hi_n        for all samples n in the window
  sum_k z_ik <= 1, z binary;    minimise number of changes

h_i is the PCM response of coefficient i from the identified linear model; only the
*change* is modelled, the baseline is the decoder's exact float output, so model
error is proportional to the (small) change.  Every accepted solution is re-checked
with the decoder's own code.
"""
import time
from functools import lru_cache

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp

from . import lattice, linmodel as LM
from .rounding import intervals, round_model
from .reconstruct import mismatches

RESP_LEN = 1152 + 512


@lru_cache(None)
def responses(c, c2, sr=44100):
    """[576, RESP_LEN] PCM response of each coefficient of a granule at offset 0."""
    A, B = LM.hybrid_blocks(sr)
    P = LM.poly_filters(sr)
    R = np.zeros((576, RESP_LEN))
    for i in range(576):
        u = np.zeros((4, 576))
        u[0] = A[c][:, i]
        u[1] = B[(c, c2)][:, i]
        y = LM.poly_synth(LM.slots_from_granules(u), P)
        R[i] = y[:RESP_LEN]
    return R


def coeff_options(ix, xhat, scale, sigma, zmax=3.0, max_opts=2):
    """Alternative ix values for one coefficient whose value is within zmax*sigma of xhat."""
    cur = lattice.dequant(ix, scale)
    opts = []
    for d in (-2, -1, 1, 2):
        v = ix + d
        if abs(v) > lattice.IX_MAX:
            continue
        val = lattice.dequant(v, scale)
        if abs(val - xhat) <= zmax * sigma + abs(cur - xhat):
            opts.append((abs(val - xhat), v, val - cur))
    opts.sort()
    return [(v, dv) for _, v, dv in opts[:max_opts]]


def _hinge(y, lo, hi):
    return np.maximum(0.0, lo - y) + np.maximum(0.0, y - hi)


def greedy_search(M, base, lo, hi, groups_of, cost=None, mu=0.02, max_steps=300):
    """Toggle search minimising interval violation (+ mu * likelihood cost). M dense [rows, nv]."""
    nv = M.shape[1]
    cost = np.zeros(nv) if cost is None else cost
    z = np.zeros(nv, bool)
    y = base.copy()
    cur = _hinge(y, lo, hi).sum()
    for _ in range(max_steps):
        if cur <= 0:
            break
        sign = np.where(z, -1.0, 1.0)
        Y = y[:, None] + M * sign[None, :]
        v = (np.maximum(0.0, lo[:, None] - Y) + np.maximum(0.0, Y - hi[:, None])).sum(axis=0)
        score = v + mu * cost * sign
        j = int(np.argmin(score))
        if v[j] >= cur - 1e-12:
            break
        if not z[j]:
            for o in groups_of[j]:
                if z[o]:
                    z[o] = False
                    y -= M[:, o]
        z[j] = not z[j]
        y += M[:, j] * (1.0 if z[j] else -1.0)
        cur = _hinge(y, lo, hi).sum()
    return z if cur <= 0 else None


def solve_window(yv, xv, rows, cands, margin=-0.006, time_limit=20.0, cost=None, N=None):
    """Find alternative values making every row land in its rounding interval.

    yv, xv: flattened (channel-major) exact float output and int16 target.
    rows: flat indices to constrain.  cands: list of (key, dv, parts) with parts =
    [(flat_start, response_vector)].  N: samples per channel (for the n%16 rounding rule).
    Returns the chosen keys or None.  margin < 0 relaxes the intervals to absorb the
    decoder's float32 re-rounding noise; the caller verifies with the exact decoder."""
    N = len(xv) if N is None else N
    lo, hi, imp = intervals(xv[rows], margin, pos=rows % N)
    if imp.any() or not cands:
        return None
    base = yv[rows]
    nv = len(cands)
    rowpos = np.full(len(yv), -1, np.int64)
    rowpos[rows] = np.arange(len(rows))
    M = np.zeros((len(rows), nv))
    for j, (_, dv, parts) in enumerate(cands):
        for st, vec in parts:
            idx = np.arange(st, min(st + len(vec), len(yv)))
            p = rowpos[idx]
            ok = p >= 0
            M[p[ok], j] += vec[:len(idx)][ok] * dv
    groups = {}
    for j, (key, _, _) in enumerate(cands):
        groups.setdefault(key[:-1], []).append(j)
    groups_of = [[o for o in groups[c[0][:-1]] if o != j] for j, c in enumerate(cands)]
    z = greedy_search(M, base, lo, hi, groups_of, cost)
    if z is not None:
        return [cands[j][0] for j in np.nonzero(z)[0]]
    # MILP fallback on rows that can possibly be violated
    reach = np.abs(M).sum(axis=1)
    active = (base - reach < lo) | (base + reach > hi)
    gr, gc = [], []
    for g, js in enumerate(groups.values()):
        gr += [g] * len(js)
        gc += js
    Gm = sp.csr_matrix((np.ones(len(gc)), (gr, gc)), shape=(len(groups), nv))
    cons = [so.LinearConstraint(sp.csr_matrix(M[active]), lo[active] - base[active], hi[active] - base[active]),
            so.LinearConstraint(Gm, -np.inf, 1.0)]
    res = so.milp(np.zeros(nv) if cost is None else cost, constraints=cons, integrality=np.ones(nv),
                  bounds=so.Bounds(0, 1), options=dict(time_limit=time_limit, disp=False))
    if res.x is None:
        return None
    return [cands[j][0] for j in np.nonzero(res.x > 0.5)[0]]


def _options_vec(ix, xhat, scale, sigma, zmax):
    """Vectorised alternative values: returns list of (d, newix, dv) arrays masks."""
    cur = lattice.dequant(ix, scale)
    base = np.abs(cur - xhat)
    out = []
    for d in (-1, 1, -2, 2):
        v = ix + d
        ok = np.abs(v) <= lattice.IX_MAX
        val = lattice.dequant(np.clip(v, -lattice.IX_MAX, lattice.IX_MAX), scale)
        ok &= np.abs(val - xhat) <= zmax * sigma + base
        ok &= scale > 0
        out.append((d, ok, v, val - cur))
    return out


ZERO = 10 ** 6       # option code meaning "set this coefficient to 0"


class Repair:
    """Causal repair sweep over granules for a reconstruction.Candidate (mono or stereo).

    For PCM granule k with mismatches, low-confidence coefficients of granules k-back..k
    (all coded channels) get alternative lattice values.  In MS granules a coded M value
    moves both outputs (+,+) and S moves them (+,-).  Each window is solved on the exact
    decoder baseline by greedy search, falling back to MILP; solutions are verified with
    the decoder and kept only if they reduce the exact mismatch count."""

    def __init__(self, cand, log=print):
        self.cand = cand
        self.log = log
        self.verbose = False
        self.milp_time = 5.0
        self.time_budget = 600.0

    def _cands(self, g0, g1, zmax, max_cands):
        cd = self.cand
        G, C, N = cd.G, cd.C, len(cd.x)
        ix = cd.recs["ix"]
        out = []
        for g in range(g0, g1 + 1):
            for c in range(C):
                t = cd.types[c][g]
                t2 = cd.types[c][g + 1] if g + 1 < G else LM.LONG
                R = responses(t, t2, cd.sr)
                if cd.ms[g] and C == 2:
                    signs = (1.0, 1.0 if c == 0 else -1.0)
                else:
                    signs = tuple(1.0 if oc == c else 0.0 for oc in range(C))
                sc, xh, sg = cd.scale_of[g, c], cd.xhat[g, c], cd.sigma[g, c]
                opts = _options_vec(ix[g, c], xh, sc, sg, zmax)
                if g >= G - 4:
                    cz = lattice.dequant(ix[g, c], sc)
                    opts.append((ZERO, ix[g, c] != 0, np.zeros_like(ix[g, c]), -cz))
                cur = lattice.dequant(ix[g, c], sc)
                for d, ok, v, dv in opts:
                    for i in np.nonzero(ok)[0]:
                        new = lattice.dequant(v[i], sc[i])
                        llr = ((xh[i] - new) ** 2 - (xh[i] - cur[i]) ** 2) / (2 * sg[i] ** 2)
                        parts = [(oc * N + 576 * g, R[i] * s) for oc, s in enumerate(signs) if s]
                        out.append(((g, c, i, d), dv[i], parts, llr))
        if len(out) > max_cands:
            out.sort(key=lambda t: t[3])
            out = out[:max_cands]
        return out

    def run(self, zmax=3.0, margin=-0.006, max_cands=2000, back=2):
        cd = self.cand
        x, G, C = cd.x, cd.G, cd.C
        N = len(x)
        xv = x.T.reshape(-1)
        y = cd.synth()
        start = mismatches(y, x)
        self.log(f"  repair: {start} mismatching samples before sweep")
        stats = dict(steps=0, changes=0, failures=[])
        t0 = time.time()
        k = 0
        while k < G:
            r0, r1 = 576 * k, min(N, 576 * (k + 1))
            if mismatches(y, x, r0, r1) == 0:
                k += 1
                continue
            if time.time() - t0 > self.time_budget:
                stats["failures"].append(k)
                stats["timed_out"] = True
                k += 1
                continue
            ok = False
            attempts = [(back, zmax, margin), (back, zmax, 0.0), (back, zmax, 0.004),
                        (back + 1, zmax + 1.5, 0.0), (back + 2, zmax + 3.0, 0.004)]
            for bk, zm, mg in attempts:
                g0 = max(0, k - bk)
                rows = np.concatenate([oc * N + np.arange(576 * g0, r1) for oc in range(C)])
                cl = self._cands(g0, k, zm, max_cands)
                cands = [c[:3] for c in cl]
                cost = np.array([c[3] for c in cl]) + 1e-3
                ts = time.time()
                sol = solve_window(y.T.reshape(-1), xv, rows, cands, mg, self.milp_time, cost, N)
                stats["steps"] += 1
                if self.verbose:
                    self.log(f"      k={k} back={bk} margin={mg} cands={len(cands)} "
                             f"sol={None if sol is None else len(sol)} {time.time() - ts:.2f}s")
                if sol is None:
                    continue
                before_ix = cd.recs["ix"].copy()
                for (g, c, i, d) in sol:
                    cd.recs["ix"][g, c][i] = 0 if d == ZERO else cd.recs["ix"][g, c][i] + d
                yt = cd.synth()
                # judge on the whole footprint of the changed granules (up to 2 granules + 512
                # samples past granule k), so later samples are never traded for these ones
                fe = min(N, 576 * (k + 3))
                before = mismatches(y, x, 576 * g0, fe)
                after = mismatches(yt, x, 576 * g0, fe)
                if after < before:
                    y = yt
                    stats["changes"] += len(sol)
                else:
                    cd.recs["ix"] = before_ix
                if mismatches(y, x, 576 * g0, r1) == 0:
                    ok = True
                    break
            if not ok:
                stats["failures"].append(k)
            if self.verbose:
                self.log(f"    granule {k}: {'ok' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
            k += 1
        final = mismatches(cd.synth(), x)
        self.log(f"  repair: {final} mismatching samples after sweep; {stats['changes']} changes, "
                 f"{stats['steps']} window solves, failed granules {stats['failures'][:20]}")
        return dict(before=start, after=final, **stats)
