"""Command line interface.

  python -m mp3inv reconstruct target.wav -o candidate.mp3 [--json stats.json]
  python -m mp3inv verify candidate.mp3 target.wav
  python -m mp3inv oracle-reencode original.mp3 -o rewritten.mp3
"""
import argparse
import json
import sys

import numpy as np

from .refdec import Harness, decode_file
from .wav import read_wav


def cmd_reconstruct(a):
    from .reconstruct import reconstruct
    x, sr = read_wav(a.target)
    mp3, stats = reconstruct(x, sr, rho_refine=a.rho_refine, shift=a.shift, repair=not a.no_repair)
    with open(a.output, "wb") as f:
        f.write(mp3)
    ok, info = verify(a.output, a.target)
    stats["verify"] = info
    print(json.dumps(info))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(stats, f, indent=1, default=int)
    return 0 if ok else 1


def verify(mp3_path, wav_path):
    """The success criterion: pristine reference decode == target, sample for sample."""
    y, sr_y = decode_file(mp3_path)
    x, sr_x = read_wav(wav_path)
    info = dict(sr_equal=sr_y == sr_x, shape_candidate=list(y.shape), shape_target=list(x.shape))
    if y.shape != x.shape:
        info.update(exact=False, mismatches=None)
        return False, info
    d = y.astype(np.int32) - x.astype(np.int32)
    info.update(exact=bool(sr_y == sr_x and not d.any()), mismatches=int(np.count_nonzero(d)),
                max_abs_diff=int(np.abs(d).max()) if d.size else 0)
    return info["exact"], info


def cmd_verify(a):
    ok, info = verify(a.candidate, a.target)
    print(json.dumps(info))
    return 0 if ok else 1


def cmd_oracle_reencode(a):
    from .bitstream import write_mp3
    data = open(a.original, "rb").read()
    frs, grs, _ = Harness("s16").dump(data)
    nch = int(frs["channels"][0])
    sr = int(frs["hz"][0])
    frames = [(int(f["mode"]), int(f["mode_ext"]), [[grs[i, gr, ch] for ch in range(nch)] for gr in range(2)])
              for i, f in enumerate(frs)]
    out = write_mp3(frames, sr, nch)
    with open(a.output, "wb") as f:
        f.write(out)
    print(f"{len(data)} -> {len(out)} bytes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mp3inv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reconstruct", help="WAV -> candidate MP3 (blind)")
    p.add_argument("target")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--json")
    p.add_argument("--rho-refine", type=float, default=16.0)
    p.add_argument("--shift", type=int, default=16)
    p.add_argument("--no-repair", action="store_true")
    p.set_defaults(fn=cmd_reconstruct)
    p = sub.add_parser("verify", help="decode candidate with the reference decoder and compare")
    p.add_argument("candidate")
    p.add_argument("target")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("oracle-reencode", help="rewrite an MP3 from its decoded parameters")
    p.add_argument("original")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_oracle_reencode)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
