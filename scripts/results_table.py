#!/usr/bin/env python3
"""Render work/results*.json as a markdown table (for docs/REPORT.md)."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "work", "results*.json")))
    res = {}
    for f in files:
        res.update(json.load(open(f)))
    print("| case | exact | samples | direct-snap mismatches | final mismatches | MS frames (ours/orig) "
          "| bands refined | orig kbps | cand kbps | time (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(res):
        r = res[name]
        if "error" in r:
            print(f"| {name} | error: {r['error'][:60]} |||||||||")
            continue
        v = r.get("verify", {})
        n = r["samples"] * r["channels"]
        d0 = r.get("mismatch_direct", 0)
        fin = v.get("mismatches")
        print(f"| {name} | {'**yes**' if v.get('exact') else 'no'} | {n} | {d0} ({100 * d0 / n:.2f}%) "
              f"| {fin} ({100 * (fin or 0) / n:.2f}%) | {r.get('ms_frames', 0)}/{r.get('orig_ms_frames', 0)} "
              f"| {r.get('refined_entries', 0)} | {r.get('orig_kbps')} | {r.get('kbps')} | {r.get('wall_s')} |")


if __name__ == "__main__":
    main()
