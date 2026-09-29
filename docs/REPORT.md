# Reconstructing an MP3 preimage from decoded PCM — report

**Question.** Given `target.wav = decode(original.mp3)` under a chosen reference decoder,
can we build *any* legal MP3 with `decode(candidate.mp3) == target.wav` for every sample,
using only the WAV?

**Short answer.**

* **Yes, for material whose quantisation is coarse relative to the 16-bit LSB.** The
  blind pipeline reconstructs a legal MP3 whose decode equals the target **on every
  sample**, verified with the unmodified reference decoder. It does so for all three
  constrained cases (mono tones, mono music, L/R stereo music, 4 s each) produced by a
  simple constant-SNR encoder. In those cases it largely recovers the original
  integers. Where it doesn't (sparse bands, file tail), it finds a *different*
  preimage.
* **Not yet for real LAME encodes.** Across LAME 3.100 at 64–320 kbps (CBR and V2),
  mono / L/R / joint stereo, at −3 and −20 dBFS, 0.4–8.7% of samples remain wrong
  (never by more than 2 LSB). No LAME case is exact.
* **The failure has a precise cause.** LAME quantises quiet high-frequency bands with
  steps comparable to the int16 rounding noise ("middle regime"). There, neither the
  scale nor the integers can be read off the PCM, and finding integers that put every
  sample inside its rounding interval is a hard lattice-decoding problem. Even with the
  original's *scales* handed over, local search fixes only ~5% of the resulting
  mismatches.
* **Everything else in the chain is solved and exact:** bitstream syntax, Huffman
  coding, reservoir, block switching, MS stereo, decoder float arithmetic and rounding
  quirks. So the remaining gap is a well-defined decoding problem, not a modelling one.

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
6. **Stereo.** For each frame, choose L/R or M/S (M=(L+R)/2, S=(L−R)/2, with the
   decoder's −2 quarter-step MS gain) by the summed MDL cost of both granules. MS needs
   equal block types in both channels. The coded channels (L,R or M,S) then go through
   the same scale/snap machinery. Synthesis and repair are joint: an M coefficient moves
   both outputs (+,+) and an S coefficient moves them (+,−).
7. **Superset refinement** (Section 4.3). Bands in the ambiguous regime get the lattice
   `q−16`, which contains the original lattice up to float rounding. Only bands with at
   least 4 nonzeros qualify, never in the tail, and only while the granule stays under
   the 4095-bit limit.
8. **End of file.** The last granules' output runs past the end of the file, so the
   inverse is ill-conditioned there. Their noise level is inflated, and the last 4
   granules are re-solved by ridge least squares on the samples that exist (MS-aware).
9. **Repair** (`mp3inv/repair.py`).
   - A causal sweep over granules, starting from the exact float output of the decoder
     for the current candidate. The model predicts the effect of changing low-confidence
     coefficients to neighbouring lattice values (or to zero, in the tail).
   - A greedy toggle search runs first, then a MILP (HiGHS) that minimises the
     log-likelihood cost subject to every sample staying inside its rounding interval.
   - Every change is re-checked with the decoder's code and kept only if it lowers the
     exact mismatch count over the changed coefficients' whole footprint.
   - Fallbacks: *scale moves* (try every scalefactor for sparse bands near a failure),
     then an *exact polish* (greedy ±1/±2/zero coordinate search where every trial is a
     real decoder run, 3.6 ms per file synthesis).
10. **Write** a legal MPEG-1 Layer III stream (`mp3inv/bitstream.py`). The writer
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
* **But the syntax limits it.** The numbers above divided the scale floats directly. In
  a real bitstream a band can only get finer through its scalefactor. Bands 11–20 have
  at most 3 bits (sf ≤ 7), so +16 quarter-steps (sf+8 at scalefac_scale 0) is usually
  impossible exactly where the middle regime lives, in the high bands. The
  alternative, lowering global_gain by 16, needs every *other* band's ix ×8 to stay
  ≤ 8206 and within the bit budget. With oracle scales, representable per-band
  refinement left 4262 mismatches vs 4147 without it (LAME tones, 128k, 2 s), and
  granule-wide `global_gain−16` applied to only 16 of 158 granules.
* **Blind, refinement barely helps** (8351 → 8104 mismatches, at +12% bitrate), because
  it needs the *correct* base lattice: a superset of a wrong lattice does not contain
  the original values.

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

### 4.5 The integer search in the middle regime is the real wall

`scripts/oracle_scales.py` separates the two blind sub-problems. It takes the
original's block types, MS flags, global gains and scalefactors, estimates only the
integers from the WAV, and runs the same repair.

| LAME tones, 128k, 2 s (91 008 samples) | mismatches |
|---|---|
| blind, direct snap | 7237 |
| oracle scales, direct snap | 4147 |
| oracle scales + full repair (greedy/MILP sweep, scale moves, exact polish) | 3933 |

So even with perfect scales, local search fixes only ~5%. The problem is lattice
decoding with a box-shaped noise model:

* each ambiguous coefficient touches ~1600 samples;
* each sample constraint is individually weak;
* the information is spread across many samples;
* windows restricted to already-seen samples admit non-original solutions that dead-end
  later, while full-footprint windows couple 3–4 granules and hundreds of binaries.

HiGHS hits its time limit (5–60 s per window) without a feasible point. Greedy toggling
stalls in local minima: a real change moves ~20 samples near a boundary, some in the
wrong direction. By contrast, sparse errors (a few wrong coefficients per granule, as in
the toy cases) are fixed reliably in ~0.1 s per granule.

### 4.6 End of file

The last ~3 granules produce output past the end of the file, so their spectra are
poorly determined and the TDAC/polyphase inverse returns garbage there. Three fixes
were needed before the constrained cases became exact:

1. ridge least squares over the samples that exist, jointly over both coded channels,
   followed by scale estimation on that estimate with the *uninflated* noise level;
2. a legal final block type (LONG/STOP). minimp3 windows the previous granule's
   overlap with the next granule's type, so this changes observed samples. LAME ends
   short→stop;
3. MS/LR in the tail frames inherited from the last reliable frame. Forcing L/R there
   is wrong when the original is M/S, because L = M+S is generally not on any single
   L lattice.

### 4.7 Joint stereo

For each frame, MS vs L/R is decided by comparing MDL costs. On a strongly correlated
LAME `-m j` clip it matched the original on all 38 decidable frames. On pure-L/R toy
stereo it chose L/R everywhere. Coded M/S values are reconstructed and repaired
jointly; the toy L/R stereo case is exact. Joint-stereo LAME material fails at the
same place as mono: the middle regime.

### 4.8 Decoder quirks that matter for exactness

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

`scripts/run_experiments.py` (2 s LAME clips, 4 s toy clips, 44.1 kHz) produces
`docs/results.json`, and `scripts/results_table.py` renders it. "Exact" is judged by
`python -m mp3inv verify`, i.e. by the pristine `build/refdec` over every sample.

Columns:
- *direct-snap* is before any search, *final* after repair.
- *MS frames* counts frames coded M/S (ours / the original's; frame 0 is LAME's Info
  frame).
- *bands refined* is the number of superset-refined bands.
- Time is wall-clock on one core, dominated by the repair search (300 s budget plus
  fallbacks).

| case | exact | samples | direct-snap mismatches | final mismatches | MS frames (ours/orig) | bands refined | orig kbps | cand kbps | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| lame_music_j128 | no | 182016 | 6388 (3.51%) | 3161 (1.74%) | 44/45 | 765 | 128.0 | 154.7 | 2058.3 |
| lame_music_m128 | no | 91008 | 9085 (9.98%) | 5629 (6.19%) | 0/0 | 1063 | 128.0 | 150.7 | 1741.8 |
| lame_music_m128_q20 | no | 91008 | 11979 (13.16%) | 7292 (8.01%) | 0/0 | 1429 | 128.0 | 139.5 | 1880.3 |
| lame_music_m192 | no | 91008 | 10970 (12.05%) | 7307 (8.03%) | 0/0 | 1019 | 192.0 | 185.9 | 1721.0 |
| lame_music_m320 | no | 91008 | 10346 (11.37%) | 7899 (8.68%) | 0/0 | 979 | 320.0 | 209.8 | 1790.2 |
| lame_music_m320_q20 | no | 91008 | 7195 (7.91%) | 5145 (5.65%) | 0/0 | 1373 | 320.0 | 161.5 | 1740.0 |
| lame_music_m64 | no | 91008 | 961 (1.06%) | 374 (0.41%) | 0/0 | 269 | 64.0 | 71.8 | 1547.5 |
| lame_music_m64_q20 | no | 91008 | 5586 (6.14%) | 2731 (3.00%) | 0/0 | 670 | 64.0 | 88.0 | 1764.4 |
| lame_music_mV2 | no | 91008 | 1639 (1.80%) | 1002 (1.10%) | 0/0 | 89 | 73.8 | 77.6 | 1392.7 |
| lame_music_s128 | no | 182016 | 2677 (1.47%) | 1294 (0.71%) | 0/0 | 676 | 128.0 | 146.2 | 2145.4 |
| lame_tones_m128 | no | 91008 | 7237 (7.95%) | 3497 (3.84%) | 0/0 | 545 | 128.0 | 156.8 | 1952.6 |
| toy_music_mono | **yes** | 178560 | 73 (0.04%) | 0 (0.00%) | 0/0 | 0 | 57.9 | 57.9 | 76.1 |
| toy_music_stereo | **yes** | 357120 | 55 (0.02%) | 0 (0.00%) | 0/0 | 0 | 70.1 | 70.1 | 144.7 |
| toy_tones_mono | **yes** | 178560 | 0 (0.00%) | 0 (0.00%) | 0/0 | 0 | 37.8 | 37.8 | 59.2 |

Observations:

* **Constrained cases are exact.** The toy music cases start with 55–73 wrong samples,
  from sparse bands whose scale is ambiguous and from the file tail. Repair, scale moves
  and the exact polish remove them. The candidate bitrate equals the original's.
* **LAME: repair roughly halves the mismatches but never reaches zero.** The best cases
  are low-rate CBR 64k (0.41%), L/R stereo 128k (0.71%) and V2 (1.1%). There most of
  the spectrum is coarse and the ambiguous bands are few.
* **The residue is almost entirely ±1 LSB.** Max |diff| is 1–2 in all final LAME
  results: the candidates are *perceptually* identical to the target; they miss only on
  rounding.
* **Harder cases:**
  - quiet material (−20 dBFS): more bands sit near the LSB;
  - high bitrates (192–320k): LAME lowers the steps of many bands into the middle
    regime.
* **Structure is recovered well.** Block types agree with the original in ~99% of
  granules. On joint stereo, MS decisions match 44 of 45 MS frames (the 45th is the
  Info frame, which carries no data).
* **Size.** Candidates are 0.5–1.4× the original bitrate. At 320k they are
  *smaller*: blind scales tend to be coarser, and our writer's table/region search is
  exhaustive.

## 6. What is not done / next steps

**Not implemented** (the obstacles are understood, but there is no code):

* **Gapless-trimmed targets / unknown frame offset.** The frame grid is assumed to
  start at sample 0, which holds for minimp3's frame API. A trimming decoder
  (minimp3_ex, ffmpeg with a LAME tag) shifts the grid by encoder delay + 529. The
  offset could be found by scanning the 1152 phases for the sharpest lattice fit, then
  writing a LAME/Info tag with matching delay/padding. minimp3's "no −1 at n%16==0"
  quirk also gives the phase mod 16 for free.
* **Clipped targets.** Saturated samples would become one-sided constraints, both in the
  inverse (masked CGLS) and in the repair intervals. `rounding.intervals` already models
  the clamp. Our test signals avoid clipping.
* **MPEG-2/2.5 LSF, intensity stereo, mixed blocks from other encoders.** The model and
  writer are partly ready: mixed blocks are identified, and the harness handles
  LSF/IS. What's missing is the LSF scalefactor syntax in the writer, and IS detection
  (constant per-band L/R ratio from the 7 `is_pos` values).
* **Another reference decoder** (mpg123, ffmpeg). The method carries over unchanged,
  but the linear model, lattice floats and rounding rule must be re-identified from that
  decoder's own code, just as was done here for minimp3.

**The research problem that remains: decoding the middle regime.**

* Needed: a real lattice decoder (sphere decoding / K-best on the near-orthogonal
  basis with box constraints, or belief propagation). It must decode jointly over the
  3–4 granules a coefficient touches, and use the exact decoder as the final check.
* Prerequisite: blind scale identification for sparse, near-LSB bands. It currently
  prefers a coarser lattice than LAME, and a wrong base lattice defeats superset
  refinement.
* Freedom that could be exploited better:
  - refining whole granules (global_gain − 16) where the bit budget allows;
  - switching scalefac_scale to reach bands 11–20;
  - accepting higher bitrates, since only the 4095-bit granule limit is hard.
