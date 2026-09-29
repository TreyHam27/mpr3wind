"""Core invariants.  Run with:  make && python -m pytest -q tests"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mp3inv import huffman, lattice, linmodel as LM, signals, toyenc  # noqa: E402
from mp3inv.bitstream import write_mp3  # noqa: E402
from mp3inv.refdec import Harness, decode_bytes  # noqa: E402
from mp3inv.rounding import round_model  # noqa: E402

HAVE_LAME = subprocess.run(["which", "lame"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def lame_case(tmp_path_factory):
    if not HAVE_LAME:
        pytest.skip("lame not installed")
    from make_case import make_case
    d = tmp_path_factory.mktemp("case")
    _, mp3, tgt = make_case(str(d), "c", "music", ["-m", "m", "-b", "128"], seconds=2)
    return open(mp3, "rb").read()


def test_codebooks_are_complete_prefix_codes():
    big, c1, _ = huffman.codebooks()
    for t, b in enumerate(big):
        if b is None:
            continue
        assert abs(sum(2.0 ** -ln for _, ln in b.values()) - 1) < 1e-12
    assert big[1][(0, 0)] == (1, 1) and big[1][(1, 1)] == (0, 3)     # ISO table 1
    for t in (0, 1):
        assert len(c1[t]) == 16


def test_harness_matches_reference_decoder(lame_case):
    _, _, pcm = Harness("s16").dump(lame_case)
    ref, _ = decode_bytes(lame_case)
    assert np.array_equal(pcm, ref)


def test_rounding_model(lame_case):
    _, _, pf = Harness("f32").dump(lame_case)
    ref, _ = decode_bytes(lame_case)
    assert np.array_equal(round_model(pf), ref)


def test_oracle_reencode_is_bit_exact(lame_case):
    frs, grs, _ = Harness("s16").dump(lame_case)
    frames = [(int(f["mode"]), int(f["mode_ext"]), [[grs[i, gr, 0]] for gr in range(2)])
              for i, f in enumerate(frs)]
    out = write_mp3(frames, int(frs["hz"][0]), 1)
    a, _ = decode_bytes(lame_case)
    b, _ = decode_bytes(out)
    assert np.array_equal(a, b)


def test_parametric_synth_matches_decoder(lame_case):
    H = Harness("f32")
    frs, grs, pf = H.dump(lame_case)
    g = grs[:, :, 0].reshape(-1)
    hdr = bytes(bytearray(frs["hdr"][1].astype(np.uint8)))
    st = H.new_state(hdr)
    y = H.synth(st, np.frombuffer(hdr * len(g), np.uint8).reshape(-1, 4), g[:, None].copy(), 1)
    assert np.array_equal(y, pf)


def test_linear_model_accuracy(lame_case):
    frs, grs, pf = Harness("f32").dump(lame_case)
    g = grs[:, :, 0].reshape(-1)
    types = np.where(g["block_type"] == 1, 0, g["block_type"])
    y = LM.synth(g["xr"].astype(np.float64), types)
    assert np.abs(y - pf[:, 0]).max() < 0.02


def test_scale_table_matches_decoder():
    S = lattice.scale_table()
    H = Harness("s16")
    for gg, ms, k in ((150, 0, 0), (200, 1, 30), (90, 0, 77)):
        assert S[gg, ms, k] == H.scale(gg, ms, k)


def test_blind_reconstruction_coarse_toy_case():
    """Constrained case: a coarse constant-SNR encoder.  Must reconstruct exactly."""
    from mp3inv.reconstruct import reconstruct
    x = signals.to_int16(signals.tones(seconds=1.0), -6)
    mp3 = toyenc.encode(x[:, 0], rho_min=16)
    target, sr = decode_bytes(mp3)
    cand, stats = reconstruct(target, sr, log=lambda *a: None)
    y, _ = decode_bytes(cand)
    assert np.array_equal(y, target)
