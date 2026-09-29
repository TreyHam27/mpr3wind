"""A deliberately simple MP3 encoder used to create *constrained* test cases.

Long blocks only, L/R (or mono), all scalefactors zero, one global_gain per
granule chosen so that every quantisation step is at least `rho_min` times the
PCM-rounding noise level in the spectral domain ("coarse regime"), and max |ix|
stays below `ix_max`.  No psychoacoustics: it is a constant-SNR quantiser.
"""
import numpy as np

from . import lattice, linmodel as LM
from .bitstream import write_mp3
from .refdec import GR_DTYPE

SIGMA_E = np.sqrt(1.0 / 12.0)


def encode(pcm, sr=44100, rho_min=16.0, ix_max=1000, bitrate="vbr"):
    pcm = np.asarray(pcm, np.float64)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    nch = pcm.shape[1]
    n = len(pcm)
    G = -(-n // 1152) * 2 + 2                    # whole frames, plus tail room
    x = np.zeros((G * 576, nch))
    x[:n] = pcm
    cpoly = np.sum(LM.poly_filters(sr) ** 2) / 32.0
    sigma = SIGMA_E / np.sqrt(9.0 * cpoly)
    q_min = int(np.ceil(4 * np.log2(rho_min * sigma)))
    S = lattice.scale_table()
    p43 = lattice.pow43_table()
    recs = []
    types = np.zeros(G, np.int64)
    for ch in range(nch):
        s = LM.poly_inverse(x[:, ch], iters=40)
        u = LM.granules_from_slots(s)
        X = LM.hybrid_adjoint(u, types) / 9.0
        r = np.zeros(G, GR_DTYPE)
        for g in range(G):
            peak = np.abs(X[g]).max()
            q = q_min
            if peak > 0:
                q = max(q, int(np.ceil(4 * np.log2(peak / p43[ix_max]))))
            gg = int(np.clip(q + 214, 0, 255))
            ix, _ = lattice.snap(X[g], S[gg, 0, 0])
            r[g]["global_gain"] = gg
            r[g]["ix"] = ix
        recs.append(r)
    mode = 3 if nch == 1 else 0
    frames = [(mode, 0, [[recs[ch][2 * f + gr] for ch in range(nch)] for gr in range(2)])
              for f in range(G // 2)]
    return write_mp3(frames, sr, nch, bitrate=bitrate)
