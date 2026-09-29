#!/usr/bin/env python3
"""Batch experiments: build test cases, reconstruct blind from the target WAV,
verify with the pristine reference decoder, and collect oracle diagnostics.

usage: run_experiments.py [--quick] [--out results.json] [CASE ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from make_case import make_case  # noqa: E402
from mp3inv import signals, toyenc  # noqa: E402
from mp3inv.__main__ import verify  # noqa: E402
from mp3inv.reconstruct import reconstruct  # noqa: E402
from mp3inv.refdec import Harness  # noqa: E402
from mp3inv.wav import read_wav  # noqa: E402

WORK = os.path.join(ROOT, "work", "exp")

# name: (kind, signal, options, level_db, seconds)
CASES = {
    # deliberately constrained: coarse constant-SNR toy encoder (our own bitstreams)
    "toy_tones_mono": ("toy", "tones", dict(rho_min=16), -6, 4),
    "toy_music_mono": ("toy", "music", dict(rho_min=16), -3, 4),
    "toy_music_stereo": ("toy", "music_st", dict(rho_min=16, ix_max=25), -3, 4),
    # LAME 3.100 (2 s clips keep the repair search affordable)
    "lame_tones_m128": ("lame", "tones", ["-m", "m", "-b", "128"], -3, 2),
    "lame_music_m64": ("lame", "music", ["-m", "m", "-b", "64"], -3, 2),
    "lame_music_m128": ("lame", "music", ["-m", "m", "-b", "128"], -3, 2),
    "lame_music_m192": ("lame", "music", ["-m", "m", "-b", "192"], -3, 2),
    "lame_music_m320": ("lame", "music", ["-m", "m", "-b", "320"], -3, 2),
    "lame_music_mV2": ("lame", "music", ["-m", "m", "-V", "2"], -3, 2),
    "lame_music_m64_q20": ("lame", "music", ["-m", "m", "-b", "64"], -20, 2),
    "lame_music_m128_q20": ("lame", "music", ["-m", "m", "-b", "128"], -20, 2),
    "lame_music_m320_q20": ("lame", "music", ["-m", "m", "-b", "320"], -20, 2),
    "lame_music_s128": ("lame", "music_st", ["-m", "s", "-b", "128"], -3, 2),
    "lame_music_j128": ("lame", "music_st", ["-m", "j", "-b", "128"], -3, 2),
}
QUICK = ["toy_tones_mono", "toy_music_mono", "lame_tones_m128", "lame_music_m128"]


def build(name):
    kind, sig, opts, level, secs = CASES[name]
    stereo = sig.endswith("_st")
    base = sig.replace("_st", "")
    if kind == "lame":
        _, mp3, tgt = make_case(WORK, name, base, opts, stereo=stereo, level=level, seconds=secs)
        return mp3, tgt
    import subprocess
    from mp3inv import BUILD
    fn = signals.SIGNALS[base]
    kw = dict(seconds=secs)
    if base in ("music", "noise"):
        kw["stereo"] = stereo
    x = signals.to_int16(fn(**kw), level)
    os.makedirs(WORK, exist_ok=True)
    mp3 = os.path.join(WORK, f"{name}.orig.mp3")
    tgt = os.path.join(WORK, f"{name}.target.wav")
    with open(mp3, "wb") as f:
        f.write(toyenc.encode(x, **opts))
    subprocess.run([os.path.join(BUILD, "refdec"), mp3, tgt], check=True, stderr=subprocess.DEVNULL)
    return mp3, tgt


def oracle_stats(mp3_path):
    data = open(mp3_path, "rb").read()
    frs, grs, _ = Harness("s16").dump(data)
    g = grs[frs["nsamples"] > 0]
    return dict(orig_bytes=len(data), orig_kbps=round(len(data) * 8 / (len(frs) * 1152 / int(frs["hz"][0])) / 1000, 1),
                orig_block_types={int(k): int(v) for k, v in zip(*np.unique(g["block_type"], return_counts=True))},
                orig_ms_frames=int(np.sum((frs["mode"] == 1) & (frs["mode_ext"] & 2 > 0))))


def run(name, log):
    mp3, tgt = build(name)
    x, sr = read_wav(tgt)
    t = time.time()
    cand, stats = reconstruct(x, sr, log=log)
    out = os.path.join(WORK, f"{name}.cand.mp3")
    with open(out, "wb") as f:
        f.write(cand)
    ok, info = verify(out, tgt)
    stats.update(verify=info, wall_s=round(time.time() - t, 1), **oracle_stats(mp3))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "results.json"))
    a = ap.parse_args()
    names = a.cases or (QUICK if a.quick else list(CASES))
    results = {}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    for n in names:
        print(f"=== {n}", flush=True)
        try:
            results[n] = run(n, log=lambda *m: print("  ", *m, flush=True))
        except Exception as e:  # noqa: BLE001
            results[n] = dict(error=repr(e))
        print(json.dumps(results[n].get("verify", results[n])), flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1, default=int)


if __name__ == "__main__":
    main()
