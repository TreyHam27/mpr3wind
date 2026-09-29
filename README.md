# mpr3wind — MP3 preimages of decoded PCM

Given a WAV produced by decoding an MP3, find **any** legal MP3 whose decode is
identical to the WAV, sample for sample, under a fixed reference decoder:

    decode(candidate.mp3) == target.wav      (every sample)

This is an analysis-by-synthesis / inverse-decoder proof of concept, not an encoder.
See [`docs/REPORT.md`](docs/REPORT.md) for the method, results, and the obstacles.

## Reference decoder

[minimp3](https://github.com/lieff/minimp3) (CC0), pinned in `third_party/minimp3`,
**unmodified**, frame API (`mp3dec_decode_frame`) over the whole file, no gapless
trimming, int16 output, built with `-O2 -ffp-contract=off` on x86-64 (SSE path).
`build/refdec` is that decoder; it is the only thing used for the success check.

## Quick start

```sh
scripts/setup_env.sh                        # lame, numpy, scipy, pytest; builds C parts
python3 scripts/make_case.py work demo music -- -m m -b 128   # source -> LAME -> target WAV
python3 -m mp3inv reconstruct work/demo.target.wav -o work/demo.cand.mp3 --json work/demo.json
python3 -m mp3inv verify work/demo.cand.mp3 work/demo.target.wav
python3 -m pytest -q tests
python3 scripts/run_experiments.py --quick  # batch runs -> work/results.json
```

`reconstruct` reads only the WAV. The original MP3 made by `make_case.py` is used only
by diagnostics (`scripts/run_experiments.py` compares against it after the fact).

## Layout

| path | role |
|---|---|
| `csrc/refdec.c` | pristine reference decoder CLI (int16 WAV; float build for analysis) |
| `csrc/harness.c` | links minimp3's own static internals: parameter dumps, synthesis from decoded parameters (no bitstream), spectra/subband injection for system identification |
| `mp3inv/huffman.py` | encoder codebooks enumerated from minimp3's decoding trees |
| `mp3inv/bitstream.py` | legal MPEG-1 Layer III writer (table/region/count1 search, bit reservoir, per-frame bitrate) |
| `mp3inv/linmodel.py` | float64 linear model of the decoder identified by impulse injection; adjoint and exact inverse |
| `mp3inv/lattice.py` | the decoder's exact dequantisation lattice (`L3_pow_43`, `L3_ldexp_q2` floats) |
| `mp3inv/scales.py` | MDL cost curves per scalefactor band; joint scalefactor/global-gain decomposition |
| `mp3inv/reconstruct.py` | blind pipeline: inverse → block types → scales → superset refinement → snap → repair → write |
| `mp3inv/repair.py` | exact-PCM repair: greedy/MILP search over alternative lattice values, verified with the decoder |
| `mp3inv/rounding.py` | minimp3's float→int16 conversion as per-sample intervals |
| `mp3inv/toyenc.py` | deliberately simple constant-SNR encoder for constrained test cases |
