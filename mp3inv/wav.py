"""Minimal 16-bit PCM WAV I/O."""
import struct
import numpy as np


def read_wav(path):
    """Return (samples int16 array shaped [n, ch], sample_rate)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("missing fmt/data chunk")
    tag, ch, sr, _, _, bits = fmt
    if tag != 1 or bits != 16:
        raise ValueError("only 16-bit PCM WAV supported")
    x = np.frombuffer(pcm[:len(pcm) // (2 * ch) * 2 * ch], dtype="<i2").reshape(-1, ch)
    return x.astype(np.int16), sr


def write_wav(path, x, sr):
    x = np.asarray(x, dtype=np.int16)
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    raw = x.astype("<i2").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
    hdr += b"data" + struct.pack("<I", len(raw))
    with open(path, "wb") as f:
        f.write(hdr + raw)
