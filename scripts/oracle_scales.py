#!/usr/bin/env python3
"""Diagnostic: how far does the integer search get if the *scales* were known?

Takes the original MP3's side information (block types, MS flags, global_gain,
scalefactors) from the oracle, estimates only the integers ix blindly from the target
WAV, optionally applies granule-wide superset refinement (global_gain - 16, i.e. every
scale / 16 exactly, ix x 8), then runs the same repair as the blind pipeline.
This separates the two obstacles: scale identification vs. the integer search.

usage: oracle_scales.py original.mp3 target.wav [--refine]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mp3inv import lattice  # noqa: E402
from mp3inv.bitstream import write_mp3  # noqa: E402
from mp3inv.reconstruct import (Analysis, Candidate, LONG, MIXED, entry_scales, _lay,  # noqa: E402
                                exact_polish, mismatches, scale_moves, tail_resolve)
from mp3inv.refdec import Harness, decode_bytes  # noqa: E402
from mp3inv.repair import Repair  # noqa: E402
from mp3inv.wav import read_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("target")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--no-repair", action="store_true")
    a = ap.parse_args()
    x, sr = read_wav(a.target)
    N, C = x.shape
    frs, grs, _ = Harness("s16").dump(open(a.original, "rb").read())
    recs = grs.reshape(-1, 2)[:, :C].copy()
    G = len(recs)
    ms = np.repeat((frs["mode"] == 1) & ((frs["mode_ext"] & 2) > 0), 2)
    bt, mx = recs["block_type"], recs["mixed_block_flag"]
    cls = np.where((bt == 2) & (mx == 1), MIXED, np.where(bt == 1, LONG, bt))
    types = [cls[:, c] for c in range(C)]
    ans = [Analysis(x[:, c], sr) for c in range(C)]
    xhat = np.zeros((G, C, 576)); sigma = np.zeros((G, C, 576)); scale_of = np.zeros((G, C, 576), np.float32)
    for g in range(G):
        for c in range(C):
            X, S = ans[c].spectra(types[c][g], types[c][g + 1] if g + 1 < G else LONG, [g])
            xhat[g, c], sigma[g, c] = X[0], S[0]
    if C == 2:        # coded channels are M/S in MS granules
        for g in np.nonzero(ms)[0]:
            L, R = xhat[g, 0].copy(), xhat[g, 1].copy()
            xhat[g, 0], xhat[g, 1] = (L + R) / 2, (L - R) / 2
            sigma[g] /= np.sqrt(2)
    refined = 0
    for g in range(G):
        for c in range(C):
            r = recs[g, c]
            if a.refine and g < G - 4 and r["global_gain"] >= 16 and np.abs(r["ix"]).max() * 8 <= lattice.IX_MAX:
                r["global_gain"] -= 16
                refined += 1
            scale_of[g, c] = entry_scales(r, *_lay(sr, types[c][g]), int(ms[g]))
            sc = scale_of[g, c]
            r["ix"] = np.where(sc > 0, lattice.snap(xhat[g, c], np.where(sc > 0, sc, 1))[0], 0)
    cand = Candidate(x, sr, recs, types, ms, xhat, sigma, scale_of)
    tail_resolve(cand)
    d0 = mismatches(cand.synth(), x)
    print(f"oracle scales, blind ix: {d0} mismatching samples of {x.size} (refined granule-channels: {refined})", flush=True)
    if not a.no_repair and d0:
        rp = Repair(cand)
        rs = rp.run()
        if rs["failures"]:
            scale_moves(cand, sorted(set(rs["failures"])))
            rs = rp.run()
        if rs["failures"]:
            exact_polish(cand, sorted(set(rs["failures"])))
    mp3 = write_mp3(cand.frames(), sr, C)
    y, _ = decode_bytes(mp3)
    print(f"final (pristine refdec): {int(np.count_nonzero(y != x))} mismatches, exact={np.array_equal(y, x)}, "
          f"{len(mp3) * 8 / (N / sr) / 1000:.0f} kbps")


if __name__ == "__main__":
    main()
