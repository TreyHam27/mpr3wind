"""minimp3's float -> int16 conversion (x86-64 SSE build), as intervals.

Samples at positions n % 16 == 0 (within each channel) are produced by the
scalar mp3d_scale_pcm; all others by SSE _mm_cvtps_epi32 (round half to even)
after clamping to [-32768, 32767].
"""
import numpy as np


def round_model(y):
    """Apply the decoder's conversion to float32 samples y [n] or [n, ch]."""
    y = np.asarray(y, np.float32)
    n = np.arange(y.shape[0]) % 16 == 0
    if y.ndim > 1:
        n = n[:, None]
    simd = np.rint(np.clip(y, -32768.0, 32767.0)).astype(np.int32)
    s = np.trunc(y + np.float32(0.5)).astype(np.int32)
    s = s - (s < 0)
    s = np.where(y >= 32766.5, 32767, np.where(y <= -32767.5, -32768, s))
    return np.where(n, s, simd).astype(np.int16)


def intervals(x, margin=0.0, pos=None):
    """Closed intervals [lo, hi] of pre-rounding values that map to target x.

    x: int16 [n] (one channel).  Boundaries are shrunk by `margin` so that
    solutions do not rely on tie-breaking or float noise.
    """
    x = np.asarray(x, np.float64)
    n = (np.arange(len(x)) if pos is None else np.asarray(pos)) % 16 == 0
    lo = x - 0.5
    hi = x + 0.5
    # scalar-path quirk: 0 covers (-1.5, 0.5); -1 is unreachable
    lo = np.where(n & (x == 0), -1.5, lo)
    impossible = n & (x == -1)
    lo = np.where(x >= 32767, 32766.5, lo)
    hi = np.where(x >= 32767, 1e9, hi)
    lo = np.where(x <= -32768, -1e9, lo)
    hi = np.where(x <= -32768, -32767.5, hi)
    return lo + margin, hi - margin, impossible
