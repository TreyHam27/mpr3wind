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


def solve_window(y, x, rows, cands, margin=-0.006, time_limit=20.0, cost=None):
    """cands: list of (key, dv, resp_start, resp).  Returns chosen keys or None.

    margin < 0 relaxes intervals to absorb the decoder's float32 rounding noise;
    the caller verifies with the exact decoder."""
    lo, hi, imp = intervals(x[rows], margin)
    if imp.any() or not cands:
        return None
    base = y[rows]
    nv = len(cands)
    r0 = rows[0]
    nrow = len(rows)
    M = np.zeros((nrow, nv))
    for j, (_, dv, st, resp) in enumerate(cands):
        a = max(st, r0)
        b = min(st + len(resp), r0 + nrow)
        if b > a:
            M[a - r0:b - r0, j] = resp[a - st:b - st] * dv
    groups = {}
    for j, (key, _, _, _) in enumerate(cands):
        groups.setdefault(key[:-1], []).append(j)
    groups_of = [[o for o in groups[c[0][:-1]] if o != j] for j, c in enumerate(cands)]
    z = greedy_search(M, base, lo, hi, groups_of, cost)
    if z is not None:
        return [cands[j][0] for j in np.nonzero(z)[0]]
    # MILP fallback on rows that can possibly be violated
    reach = np.abs(M).sum(axis=1)
    active = (base - reach < lo) | (base + reach > hi)
    Ma = sp.csr_matrix(M[active])
    gr, gc = [], []
    for g, js in enumerate(groups.values()):
        gr += [g] * len(js)
        gc += js
    Gm = sp.csr_matrix((np.ones(len(gc)), (gr, gc)), shape=(len(groups), nv))
    cons = [so.LinearConstraint(Ma, lo[active] - base[active], hi[active] - base[active]),
            so.LinearConstraint(Gm, -np.inf, 1.0)]
    res = so.milp(np.zeros(nv) if cost is None else cost, constraints=cons, integrality=np.ones(nv), bounds=so.Bounds(0, 1),
                  options=dict(time_limit=time_limit, disp=False))
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


class MonoRepair:
    """Repair driver for one independent channel (mono, or LR stereo channel)."""

    def __init__(self, x, types, xhat, sigma, scale_of, synth_float, sr=44100, log=print):
        self.x = x                      # int16 [N]
        self.types = types
        self.xhat, self.sigma, self.scale_of = xhat, sigma, scale_of   # [G, 576]
        self.synth_float = synth_float  # callable(ix [G,576]) -> float y [N]
        self.sr = sr
        self.log = log
        self.G = len(types)
        self.verbose = False
        self.milp_time = 5.0
        self.lookahead = 0
        self.time_budget = 600.0       # seconds for the whole sweep

    def _cands(self, ix, g0, g1, zmax, max_cands):
        cands = []
        for g in range(g0, g1 + 1):
            c = self.types[g]
            c2 = self.types[g + 1] if g + 1 < self.G else LM.LONG
            R = responses(c, c2, self.sr)
            for d, ok, v, dv in _options_vec(ix[g], self.xhat[g], self.scale_of[g], self.sigma[g], zmax):
                cur = lattice.dequant(ix[g], self.scale_of[g])
                for i in np.nonzero(ok)[0]:
                    new = lattice.dequant(v[i], self.scale_of[g, i])
                    llr = ((self.xhat[g, i] - new) ** 2 - (self.xhat[g, i] - cur[i]) ** 2) / (2 * self.sigma[g, i] ** 2)
                    cands.append(((g, i, d), dv[i], 576 * g, R[i], llr))
        if len(cands) > max_cands:
            cands.sort(key=lambda t: t[4])
            cands = cands[:max_cands]
        return cands

    def run(self, ix, zmax=3.0, margin=-0.006, max_cands=2000, back=2):
        """Causal sweep: make PCM granule k exact by changing coefficient granules k-back..k."""
        x, G = self.x, self.G
        N = len(x)
        y = self.synth_float(ix)
        start = int(np.count_nonzero(round_model(y) != x))
        self.log(f"  repair: {start} mismatching samples before sweep")
        stats = dict(steps=0, changes=0, failures=[])
        t0 = time.time()
        k = 0
        while k < G:
            r0, r1 = 576 * k, min(N, 576 * (k + 1))
            if np.array_equal(round_model(y[r0:r1]), x[r0:r1]):
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
                g1 = min(G - 1, k + self.lookahead)
                rows = np.arange(576 * g0, min(N, 576 * (g1 + 1)))
                cl = self._cands(ix, g0, g1, zm, max_cands)
                cands = [c[:4] for c in cl]
                cost = np.array([c[4] for c in cl]) + 1e-3
                ts = time.time()
                sol = solve_window(y, x, rows, cands, mg, time_limit=self.milp_time, cost=cost)
                stats["steps"] += 1
                if self.verbose:
                    self.log(f"      k={k} back={bk} margin={mg} cands={len(cands)} "
                             f"sol={None if sol is None else len(sol)} {time.time() - ts:.2f}s")
                if sol is None:
                    continue
                trial = ix.copy()
                for (g, i, d) in sol:
                    trial[g, i] += d
                yt = self.synth_float(trial)
                re = rows[-1] + 1
                before = np.count_nonzero(round_model(y[576 * g0:re]) != x[576 * g0:re])
                after = np.count_nonzero(round_model(yt[576 * g0:re]) != x[576 * g0:re])
                if after < before:           # keep progress; exact baseline for next round
                    ix, y = trial, yt
                    stats["changes"] += len(sol)
                if after == 0:
                    ok = True
                    break
            if not ok:
                stats["failures"].append(k)
            if self.verbose:
                self.log(f"    granule {k}: {'ok' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
            k += 1
        final = int(np.count_nonzero(round_model(y) != x))
        self.log(f"  repair: {final} mismatching samples after sweep; {stats['changes']} changes, "
                 f"{stats['steps']} MILPs, failed granules {stats['failures'][:20]}")
        return ix, dict(before=start, after=final, **stats)
