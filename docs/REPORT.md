# Reconstructing an MP3 preimage from decoded PCM — report

**Question.** Given `target.wav = decode(original.mp3)` under a chosen reference decoder,
can we build *any* legal MP3 with `decode(candidate.mp3) == target.wav` for every sample,
using only the WAV?

**Short answer.** *(filled in from the results section)*

---

## 1. Setup

* **Reference decoder**: minimp3 (lieff/minimp3 @ `ea99364`, CC0), unmodified, frame API
  over the whole file, no gapless trimming, int16 output, x86-64 SSE build with
  `-O2 -ffp-contract=off`. `build/refdec` is the pristine decoder; all success claims
  come from it.
* **Test material**: deterministic synthetic signals (`mp3inv/signals.py`: a small
  "song" with melody, chords, bass and drums; stationary tones; pink noise), encoded with
  LAME 3.100, and a deliberately simple coarse encoder (`mp3inv/toyenc.py`).
  The original MP3 is used only for diagnostics after reconstruction.

## 2. Method

MP3 decoding after Huffman decoding is a fixed **linear** map followed by a single
nonlinearity (float→int16 rounding with clipping):

```
ix (integers) --dequant--> xr = ±pow43(|ix|)·scale --MS--> --reorder/antialias/IMDCT/overlap-->
     subband samples --polyphase synthesis--> y (float) --round/clip--> PCM
```

The spectral values of any MP3 lie on a sparse, decoder-specific **lattice**
`±pow43_dec(n)·F(global_gain, scalefactor)`. Reconstruction is therefore a lattice
problem: find integers and scales whose synthesis lands every sample inside its
rounding interval.

Pipeline (`mp3inv/reconstruct.py`):

1. **Exact linear model by system identification.** The polyphase filters (32×512 taps)
   and the hybrid-filterbank blocks for every window-type pair are measured by injecting
   impulses into minimp3's own code (`csrc/harness.c`). Model vs. decoder: max error
   0.008 LSB (rms 4e-4 LSB) on real files.
2. **Inverse.** CGLS on the polyphase stage (near-orthogonal; 40 iterations, residual
   4e-9 LSB), then TDAC analysis per granule. From int16 PCM the spectra are recovered to
   σ ≈ 2.6e-7 (in minimp3 `xr` units): the int16 rounding noise
   (1/12 LSB² per sample) projected into the spectral domain.
3. **Block types** by Viterbi over legal window sequences (long→start→short…→stop→long),
   scoring each hypothesis with the MDL cost of the best lattice fit.
4. **Scales.** Per scalefactor band, an MDL cost curve over every quarter-step exponent
   `q` (Gaussian likelihood of the residuals under noise σ, plus a bit-cost prior),
   followed by a joint search over (global_gain, scalefac_scale, preflag, subblock_gain,
   scalefactors) that minimises the summed cost under the real syntax constraints.
   Exact scale floats come from the decoder's own `L3_ldexp_q2`.
5. **Snap** the spectra to the decoder's own lattice (minimp3's `L3_pow_43`, including
   its polynomial approximation above 128).
6. **Superset refinement** (Section 4.3). Bands in the ambiguous regime get the lattice
   `q−16`, which contains the original lattice up to float rounding.
7. **Repair** (`mp3inv/repair.py`). This is a causal sweep over granules. It uses the
   exact float output of the decoder for the current candidate. The model predicts the
   effect of changing low-confidence coefficients to neighbouring lattice values. A
   greedy toggle search runs first, then a MILP (HiGHS) that minimises the
   log-likelihood cost subject to every sample staying inside its rounding interval.
   Each accepted change is re-checked with the decoder's code.
8. **Write** a legal MPEG-1 Layer III stream (`mp3inv/bitstream.py`). The writer
   chooses Huffman tables, region split, big_values/count1 boundary, count1 table,
   scalefac_compress, reservoir usage and per-frame bitrate to minimise size.

## 3. Things that work (verified)

* **Bit-exact harness.** Parametric synthesis from decoded parameters (no bitstream) is
  identical to the reference decoder on every sample, in both int16 and float. The
  rounding model below reproduces every sample.
* **Codebooks from the decoder.** All 29 big-value tables and both count1 tables were
  enumerated from minimp3's lookup trees. Each is a complete prefix code (Kraft sum = 1)
  and matches the ISO tables.
* **Oracle re-encode.** Taking the decoded parameters of a LAME file and writing them with
  our own writer (different tables, regions, reservoir use and bitrates) gives a file
  that decodes **bit-exactly** to the same PCM. The VBR version is slightly smaller than
  LAME's original (63.9 kB vs 65.2 kB). Only the decoded parameters matter; everything
  else in the bitstream is free.
* **Coarse-regime recovery.** When a coefficient's quantisation step is at least ~8×
  the spectral noise σ, blind snapping recovers the *original* integer in 100% of cases
  (tens of thousands of coefficients, zero errors). Block types agree with the
  original in ~99% of granules.

## 4. The main obstacles

### 4.1 The three regimes

Let ρ = (lattice step)/σ for a coefficient.

* **Coarse (ρ ≳ 8):** snapping returns the original integer. The PCM rounding offsets
  of the original are then reproduced exactly as well.
* **Fine (ρ ≲ 1):** many lattice points fit, and least squares plus snapping stays
  inside the rounding box.
* **Middle (ρ ≈ 2–8):** neither holds. Snapping errors appear at a rate of ~1–7% of
  coefficients.

A key subtlety: *mixing* regimes is harmful. Let `X̂ = S⁻¹x` be the least-squares
spectrum and `e` the (unknown) rounding error of the original. Coarse coefficients
snap to the truth, and so reproduce the projection `P_c e` of the rounding error onto
the coarse subspace. That projection is **not** inside the ±0.5 box, so the remaining
coefficients must compensate exactly. When everything is coarse, `P_c e = e` is inside
the box. When everything is fine, `P_c e ≈ 0`.

### 4.2 The encoder puts coefficients in the middle regime on purpose

LAME's absolute threshold of hearing puts the quantisation noise of quiet
high-frequency bands at roughly the 16-bit LSB level, which is exactly ρ ≈ 1–8.
Measured share of nonzero coefficients by ρ, and the PCM mismatches left when the
*true* scales are known and only the integers are estimated (oracle scales):

| material | <2 | 2–4 | 4–8 | 8–32 | ≥32 | mismatching samples |
|---|---|---|---|---|---|---|
| music −3 dBFS, 64k | 0.00 | 0.00 | 0.09 | 0.20 | 0.71 | 0.52% |
| music −3 dBFS, 128k | 0.25 | 0.18 | 0.08 | 0.07 | 0.42 | 1.81% |
| music −3 dBFS, 192k | 0.57 | 0.05 | 0.04 | 0.08 | 0.28 | 0.51% |
| music −3 dBFS, 320k | 0.65 | 0.05 | 0.07 | 0.09 | 0.15 | 0.89% |
| music −20 dBFS, 64k | 0.01 | 0.13 | 0.22 | 0.31 | 0.33 | 2.53% |
| music −20 dBFS, 128k | 0.51 | 0.11 | 0.07 | 0.10 | 0.22 | 0.89% |
| music −20 dBFS, 192k | 0.70 | 0.05 | 0.05 | 0.09 | 0.12 | 0.46% |
| music −20 dBFS, 320k | 0.84 | 0.04 | 0.04 | 0.06 | 0.02 | 0.53% |

Every setting leaves some ambiguity. Scale estimation is also unreliable exactly
there: when ρ < 4, MDL prefers a coarser lattice than the original (dq = +2…+10 quarter
steps), and 12–18% of bands with content get a scale different from LAME's.

### 4.3 Why "just quantise finer" does not work, and what does

* The decoder lattice is `|ix|^{4/3}·s` with `|ix| ≤ 8206`. For a coefficient of size V,
  a step ≤ σ needs `ix ≈ 1.33·V/σ`, so anything louder than ~6000σ can never be fine.
  Uniformly finer lattices do not even contain the original values of loud coefficients.
  Snapping everything to a uniform lattice with ρ = 1 gave 177k/180k wrong samples.
* The bit budget is hard: `part2_3_length` ≤ 4095 bits per granule and channel.
  Near-lossless precision for all 576 coefficients would need far more.
* **Superset refinement works.** `q−16` multiplies ix by 8, and
  `pow43(8n) ≈ 16·pow43(n)` (equal up to table rounding), so the refined lattice
  contains the original values while adding 7 new points between each pair. With oracle
  scales, refining every representable band by 16 quarter-steps cut mismatches from
  8552 to 314 (tones, 128k), and by 32 to 55. Non-superset shifts (4, 8, 12, 20, 24)
  made things far worse (10k–134k mismatches). Refining only bands with ρ < 16 is the
  best trade-off: 763 / 400 mismatches at about 2× the original Huffman bits.

### 4.4 Float arithmetic makes "any preimage" nearly as hard as "the original"

minimp3 computes in float32. Changing any coefficient re-rounds every partial sum in
its ~1600-sample footprint, which adds up to ~0.004 LSB of unpredictable error at
|y| ≈ 20000. The linear model is exact up to that noise, but samples whose true value
lies within ~0.004 of a rounding boundary flip unpredictably whenever a solution
differs from the original. A 1-ulp change affects only 0.03 samples on average; a real
change of ~0.6 LSB flips a few. So:

* the *original* values are privileged: they are the only candidate that reproduces the
  decoder's float path bit for bit;
* any alternative solution must be confirmed with the exact decoder, and near-boundary
  samples turn the last steps into a search with exact evaluation.

### 4.5 Decoder quirks that matter for exactness

* `L3_pow_43` uses a polynomial approximation for |ix| ≥ 129. The lattice is the
  decoder's lattice, not `x^{4/3}`.
* Scale floats come from chained `L3_ldexp_q2` calls. 31% of (global_gain, k) pairs
  differ in the last bit from another decomposition with the same exponent.
* Rounding depends on position. Samples with `n % 16 == 0` use scalar `mp3d_scale_pcm`,
  which maps (−1.5, −0.5) to 0, so **the value −1 never occurs at those positions**
  (confirmed on all targets; also a fingerprint of this decoder). All other samples use
  SSE round-half-even.
* Windowing is applied at the *next* granule. Start blocks (type 1) decode identically
  to normal blocks, and minimp3 does not care about ISO window sequencing. We still emit
  legal sequences.

## 5. Results

*(generated by `scripts/run_experiments.py`; see below)*

## 6. What is not done / next steps

*(filled in at the end)*
