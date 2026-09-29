"""Bindings to the reference decoder (pristine minimp3) and the analysis harness.

* ``decode_file``/``decode_bytes``: run the pristine ``build/refdec`` binary.  This is
  the only thing used for the final success check.
* ``Harness``: ctypes wrapper around ``libharness_{s16,f32}.so`` (same minimp3 code)
  for parameter dumps and fast parametric synthesis.
"""
import ctypes
import os
import subprocess
import tempfile

import numpy as np

from . import BUILD
from .wav import read_wav

GR_DTYPE = np.dtype([
    ("part_23_length", "<i4"), ("big_values", "<i4"), ("global_gain", "<i4"),
    ("scalefac_compress", "<i4"), ("block_type", "<i4"), ("mixed_block_flag", "<i4"),
    ("table_select", "<i4", 3), ("region_count", "<i4", 3), ("subblock_gain", "<i4", 3),
    ("preflag", "<i4"), ("scalefac_scale", "<i4"), ("count1_table", "<i4"), ("scfsi", "<i4"),
    ("n_long_sfb", "<i4"), ("n_short_sfb", "<i4"),
    ("iscf", "<i4", 40), ("ix", "<i4", 576), ("xr", "<f4", 576), ("scf", "<f4", 40),
])
FR_DTYPE = np.dtype([
    ("offset", "<i4"), ("frame_bytes", "<i4"), ("channels", "<i4"), ("hz", "<i4"),
    ("bitrate_kbps", "<i4"), ("layer", "<i4"), ("mpeg1", "<i4"), ("mode", "<i4"),
    ("mode_ext", "<i4"), ("main_data_begin", "<i4"), ("nsamples", "<i4"), ("sr_idx", "<i4"),
    ("hdr", "<i4", 4),
])


def decode_file(mp3_path):
    """Decode with the pristine reference decoder. Returns (int16 [n, ch], sr)."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.wav")
        subprocess.run([os.path.join(BUILD, "refdec"), mp3_path, out], check=True,
                       stderr=subprocess.DEVNULL)
        return read_wav(out)


def decode_bytes(data):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.mp3")
        with open(p, "wb") as f:
            f.write(data)
        return decode_file(p)


def decode_file_f32(mp3_path, channels):
    """Pre-rounding float output (x32768) from the float build of the reference decoder."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.f32")
        subprocess.run([os.path.join(BUILD, "refdec_f32"), mp3_path, out], check=True,
                       stderr=subprocess.DEVNULL)
        x = np.fromfile(out, dtype="<f4")
    return (x.astype(np.float64) * 32768.0).reshape(-1, channels)


class Harness:
    """One instance per output flavour ('s16' or 'f32')."""

    _cache = {}

    def __new__(cls, flavour="s16"):
        if flavour in cls._cache:
            return cls._cache[flavour]
        self = super().__new__(cls)
        lib = ctypes.CDLL(os.path.join(BUILD, f"libharness_{flavour}.so"))
        self.lib, self.flavour = lib, flavour
        self.is_float = bool(lib.h_is_float())
        assert lib.h_sizeof_gr() == GR_DTYPE.itemsize, (lib.h_sizeof_gr(), GR_DTYPE.itemsize)
        assert lib.h_sizeof_fr() == FR_DTYPE.itemsize
        self.dec_size = lib.h_sizeof_dec()
        lib.h_pow43.restype = ctypes.c_float
        lib.h_pow43.argtypes = [ctypes.c_int]
        lib.h_scale.restype = ctypes.c_float
        lib.h_scale.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.sample_dtype = np.float32 if self.is_float else np.int16
        cls._cache[flavour] = self
        return self

    # ------------------------------------------------------------------ dump
    def dump(self, data, max_frames=None):
        """Decode an MP3 byte string, returning (frames, grans, pcm).

        grans has shape [nframes, 2 (granule), 2 (channel)] of GR_DTYPE.
        pcm is int16 (s16) or float32*32768 (f32) shaped [n, ch].
        """
        if max_frames is None:
            max_frames = len(data) // 48 + 16
        frs = np.zeros(max_frames, FR_DTYPE)
        grs = np.zeros(max_frames * 4, GR_DTYPE)
        max_samples = max_frames * 1152 * 2
        pcm = np.zeros(max_samples, self.sample_dtype)
        nout = ctypes.c_int(0)
        buf = ctypes.create_string_buffer(bytes(data), len(data))
        n = self.lib.h_decode_dump(buf, len(data), frs.ctypes.data_as(ctypes.c_void_p),
                                   grs.ctypes.data_as(ctypes.c_void_p), max_frames,
                                   pcm.ctypes.data_as(ctypes.c_void_p), max_samples,
                                   ctypes.byref(nout))
        frs = frs[:n]
        grs = grs[:n * 4].reshape(n, 2, 2)
        ch = int(frs["channels"][frs["nsamples"] > 0][0]) if n else 1
        pcm = pcm[:nout.value].reshape(-1, ch)
        if self.is_float:
            pcm = pcm.astype(np.float64) * 32768.0
        return frs, grs, pcm

    # ----------------------------------------------------------------- synth
    def new_state(self, hdr4):
        st = ctypes.create_string_buffer(self.dec_size)
        self.lib.h_dec_init(st, bytes(bytearray(hdr4)))
        return st

    @staticmethod
    def copy_state(st):
        new = ctypes.create_string_buffer(len(st.raw))
        ctypes.memmove(new, st, len(st.raw))
        return new

    def synth(self, state, hdrs, grans, nch, want_sb=False):
        """Parametric synthesis.  grans: GR_DTYPE array [ngr, nch]; hdrs: uint8 [ngr, 4].

        Advances ``state``.  Returns pcm [ngr*576, nch] (int16, or float*32768)."""
        grans = np.ascontiguousarray(grans)
        ngr = grans.shape[0]
        hdrs = np.ascontiguousarray(hdrs, dtype=np.uint8)
        pcm = np.zeros(ngr * 576 * nch, self.sample_dtype)
        sb = np.zeros(ngr * 576 * nch, np.float32) if want_sb else None
        r = self.lib.h_synth(state, hdrs.ctypes.data_as(ctypes.c_void_p),
                             grans.ctypes.data_as(ctypes.c_void_p), ngr, nch,
                             pcm.ctypes.data_as(ctypes.c_void_p),
                             sb.ctypes.data_as(ctypes.c_void_p) if want_sb else None)
        if r:
            raise RuntimeError(f"h_synth failed ({r})")
        pcm = pcm.reshape(-1, nch)
        if self.is_float:
            pcm = pcm.astype(np.float64) * 32768.0
        if want_sb:
            return pcm, sb.reshape(ngr, nch, 32, 18)
        return pcm

    def synth_xr(self, state, hdrs, xr, bt, mx, want_sb=False):
        """xr float32 [ngr, nch, 576]; bt/mx int [ngr, nch]."""
        xr = np.ascontiguousarray(xr, dtype=np.float32)
        ngr, nch = xr.shape[:2]
        bt = np.ascontiguousarray(bt, dtype=np.int32)
        mx = np.ascontiguousarray(mx, dtype=np.int32)
        hdrs = np.ascontiguousarray(hdrs, dtype=np.uint8)
        pcm = np.zeros(ngr * 576 * nch, self.sample_dtype)
        sb = np.zeros(ngr * 576 * nch, np.float32)
        r = self.lib.h_synth_xr(state, hdrs.ctypes.data_as(ctypes.c_void_p),
                                xr.ctypes.data_as(ctypes.c_void_p),
                                bt.ctypes.data_as(ctypes.c_void_p), mx.ctypes.data_as(ctypes.c_void_p),
                                ngr, nch, pcm.ctypes.data_as(ctypes.c_void_p),
                                sb.ctypes.data_as(ctypes.c_void_p))
        if r:
            raise RuntimeError(f"h_synth_xr failed ({r})")
        pcm = pcm.reshape(-1, nch)
        if self.is_float:
            pcm = pcm.astype(np.float64) * 32768.0
        if want_sb:
            return pcm, sb.reshape(ngr, nch, 32, 18)
        return pcm

    def poly(self, state, sb, nch):
        """sb float32 [ngr, nch, 576] -> pcm (polyphase stage only)."""
        sb = np.ascontiguousarray(sb, dtype=np.float32)
        ngr = sb.shape[0]
        pcm = np.zeros(ngr * 576 * nch, self.sample_dtype)
        self.lib.h_poly(state, sb.ctypes.data_as(ctypes.c_void_p), ngr, nch,
                        pcm.ctypes.data_as(ctypes.c_void_p))
        pcm = pcm.reshape(-1, nch)
        if self.is_float:
            pcm = pcm.astype(np.float64) * 32768.0
        return pcm

    def sfbtab(self, hdr4, block_type, mixed=0):
        w = np.zeros(40, np.int32)
        nl, ns = ctypes.c_int32(), ctypes.c_int32()
        n = self.lib.h_sfbtab(bytes(bytearray(hdr4)), block_type, mixed,
                              w.ctypes.data_as(ctypes.c_void_p), ctypes.byref(nl), ctypes.byref(ns))
        return w[:n].copy(), nl.value, ns.value

    def pow43(self, x):
        return self.lib.h_pow43(int(x))

    def scale(self, gg, ms, k):
        return self.lib.h_scale(int(gg), int(ms), int(k))
