#!/usr/bin/env python3
"""Create one test case: synthetic source -> LAME MP3 -> reference-decoded target WAV.

usage: make_case.py OUTDIR NAME SIGNAL [--stereo] [--level DB] [--seconds S] -- LAME_OPTS...
Writes OUTDIR/NAME.src.wav, NAME.orig.mp3 (development oracle only), NAME.target.wav
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mp3inv import BUILD, signals  # noqa: E402
from mp3inv.wav import write_wav  # noqa: E402


def make_case(outdir, name, signal, lame_opts, stereo=False, level=-3.0, seconds=None, sr=44100, seed=None):
    os.makedirs(outdir, exist_ok=True)
    fn = signals.SIGNALS[signal]
    kw = {"sr": sr}
    if seconds:
        kw["seconds"] = seconds
    if seed is not None:
        kw["seed"] = seed
    if signal in ("music", "noise"):
        kw["stereo"] = stereo
    x = fn(**kw)
    if stereo and x.shape[1] == 1:
        x = x.repeat(2, axis=1)
    src = os.path.join(outdir, f"{name}.src.wav")
    mp3 = os.path.join(outdir, f"{name}.orig.mp3")
    tgt = os.path.join(outdir, f"{name}.target.wav")
    write_wav(src, signals.to_int16(x, level), sr)
    subprocess.run(["lame", "--quiet", *lame_opts, src, mp3], check=True)
    subprocess.run([os.path.join(BUILD, "refdec"), mp3, tgt], check=True, stderr=subprocess.DEVNULL)
    return src, mp3, tgt


if __name__ == "__main__":
    argv = sys.argv[1:]
    lame = []
    if "--" in argv:
        i = argv.index("--")
        argv, lame = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("name")
    ap.add_argument("signal", choices=sorted(signals.SIGNALS))
    ap.add_argument("--stereo", action="store_true")
    ap.add_argument("--level", type=float, default=-3.0)
    ap.add_argument("--seconds", type=float)
    a = ap.parse_args(argv)
    for p in make_case(a.outdir, a.name, a.signal, lame, a.stereo, a.level, a.seconds):
        print(p)
