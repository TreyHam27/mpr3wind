"""Deterministic synthetic test signals (no external audio needed)."""
import numpy as np


def _env(n, sr, a=0.01, d=0.2, s=0.6, r=0.3):
    t = np.arange(n) / sr
    e = np.interp(t, [0, a, a + d, max(a + d, n / sr - r), n / sr], [0, 1, s, s, 0])
    return e


def _note(freq, dur, sr, rng, bright=1.0):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for h in range(1, 12):
        if freq * h > sr / 2 - 500:
            break
        y += (bright ** (h - 1)) / h * np.sin(2 * np.pi * freq * h * t * (1 + 0.0005 * h) + rng.uniform(0, 6.28))
    vib = 1 + 0.003 * np.sin(2 * np.pi * 5.5 * t)
    y = np.interp(np.cumsum(vib) - vib[0], np.arange(n), y)
    return y * _env(n, sr)


def _drum(sr, rng, kind):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    if kind == "kick":
        f = 60 + 90 * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 12)
    if kind == "snare":
        return (rng.standard_normal(n) * 0.7 + np.sin(2 * np.pi * 190 * t) * 0.5) * np.exp(-t * 25)
    hh = rng.standard_normal(n)
    hh = np.diff(hh, prepend=0)  # crude high-pass
    return hh * np.exp(-t * 60) * 0.5


def music(seconds=8.0, sr=44100, stereo=False, seed=1):
    """Melody + chords + bass + drums; transients trigger short blocks."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros((n, 2))
    beat = 60 / 112
    scale = [0, 2, 3, 5, 7, 8, 10]
    t0 = 0.0
    while t0 < seconds - 0.5:
        # melody
        deg = rng.choice(scale) + 12 * rng.integers(0, 2)
        f = 220 * 2 ** (deg / 12)
        dur = beat * rng.choice([0.5, 1, 1, 2])
        y = _note(f, dur * 1.1, sr, rng, bright=0.7) * 0.25
        i = int(t0 * sr)
        pan = rng.uniform(0.2, 0.8)
        m = min(len(y), n - i)
        out[i:i + m, 0] += y[:m] * (1 - pan)
        out[i:i + m, 1] += y[:m] * pan
        t0 += dur
    for k in range(int(seconds / beat)):
        i = int(k * beat * sr)
        if k % 4 == 0:  # chord + bass each bar
            root = 110 * 2 ** (rng.choice([0, 5, 7, 3]) / 12)
            for iv in (1, 1.26, 1.5):
                y = _note(root * 2 * iv, beat * 4, sr, rng, bright=0.5) * 0.08
                m = min(len(y), n - i)
                out[i:i + m] += y[:m, None] * [0.6, 0.4]
            y = _note(root / 2, beat * 4, sr, rng, bright=0.3) * 0.3
            m = min(len(y), n - i)
            out[i:i + m] += y[:m, None] * 0.5
        for kind, cond in (("kick", k % 2 == 0), ("snare", k % 2 == 1), ("hat", True)):
            if cond:
                y = _drum(sr, rng, kind) * (0.5 if kind != "hat" else 0.3)
                m = min(len(y), n - i)
                pan = 0.5 if kind != "hat" else 0.7
                out[i:i + m, 0] += y[:m] * (1 - pan)
                out[i:i + m, 1] += y[:m] * pan
    out /= np.max(np.abs(out)) + 1e-9
    return out if stereo else out.mean(axis=1, keepdims=True) / np.max(np.abs(out.mean(axis=1))) 


def tones(seconds=6.0, sr=44100, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    for f in (220, 330, 440, 1234, 3000, 7500):
        y += rng.uniform(0.3, 1) * np.sin(2 * np.pi * f * t * (1 + 0.001 * np.sin(2 * np.pi * 0.3 * t)))
    return (y / np.max(np.abs(y)))[:, None]


def noise(seconds=5.0, sr=44100, seed=3, stereo=False):
    rng = np.random.default_rng(seed)
    ch = 2 if stereo else 1
    w = rng.standard_normal((int(seconds * sr), ch))
    # pink-ish: cumulative filter
    f = np.fft.rfft(w, axis=0)
    k = np.arange(f.shape[0])[:, None]
    f /= np.sqrt(np.maximum(k, 1))
    y = np.fft.irfft(f, n=w.shape[0], axis=0)
    y *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * np.arange(w.shape[0]) / sr))[:, None]
    return y / np.max(np.abs(y))


def to_int16(x, level_db=-3.0):
    g = 10 ** (level_db / 20) * 32767
    return np.clip(np.round(x * g), -32768, 32767).astype(np.int16)


SIGNALS = {"music": music, "tones": tones, "noise": noise}
