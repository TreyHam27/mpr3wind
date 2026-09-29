/*
 * harness: analysis/synthesis entry points built on the *unmodified* minimp3
 * implementation.  Because minimp3's internals are static functions in the
 * same translation unit, they can be called directly -- no patching.
 *
 * Built twice: libharness_s16.so (int16 output, identical to refdec) and
 * libharness_f32.so (MINIMP3_FLOAT_OUTPUT: pre-rounding samples / 32768).
 *
 *  h_decode_dump   bitstream -> per-frame info, per-granule side info,
 *                  scalefactors, derived ix[], dequantised xr[], PCM
 *  h_synth         decoded parameters (no bitstream) -> PCM, running the
 *                  exact same float code as the decoder after Huffman decoding
 *  h_synth_xr      dequantised spectra -> PCM (+ subband samples); used for
 *                  system identification of the linear model
 */
#define MINIMP3_IMPLEMENTATION
#include "../third_party/minimp3/minimp3.h"
#include <string.h>
#include <math.h>

typedef struct {
    int32_t part_23_length, big_values, global_gain, scalefac_compress;
    int32_t block_type, mixed_block_flag, table_select[3], region_count[3], subblock_gain[3];
    int32_t preflag, scalefac_scale, count1_table, scfsi;
    int32_t n_long_sfb, n_short_sfb;
    int32_t iscf[40];     /* scalefactors as transmitted (after scfsi copy) */
    int32_t ix[576];      /* signed quantised values, bitstream order */
    float xr[576];        /* dequantised values, before stereo processing */
    float scf[40];        /* per sfb-table-entry scale floats */
} hgr_t;

typedef struct {
    int32_t offset, frame_bytes, channels, hz, bitrate_kbps, layer;
    int32_t mpeg1, mode, mode_ext, main_data_begin, nsamples, sr_idx;
    int32_t hdr[4];
} hfr_t;

int h_sizeof_dec(void) { return (int)sizeof(mp3dec_t); }
int h_sizeof_gr(void) { return (int)sizeof(hgr_t); }
int h_sizeof_fr(void) { return (int)sizeof(hfr_t); }
int h_is_float(void)
{
#ifdef MINIMP3_FLOAT_OUTPUT
    return 1;
#else
    return 0;
#endif
}

/* decoder's own pow43 (exposed for building the exact lattice) */
float h_pow43(int x) { return L3_pow_43(x); }
float h_ldexp_q2(float y, int e) { return L3_ldexp_q2(y, e); }

/* exact per-band scale float as computed in L3_decode_scalefactors */
static float scale_gain(int global_gain, int ms)
{
    int gain_exp = global_gain + BITS_DEQUANTIZER_OUT*4 - 210 - (ms ? 2 : 0);
    return L3_ldexp_q2(1 << (MAX_SCFI/4), MAX_SCFI - gain_exp);
}
float h_scale(int global_gain, int ms, int k) { return L3_ldexp_q2(scale_gain(global_gain, ms), k); }


/* ---------------------------------------------------------------- dump -- */

static int ix_from_float(float v, float one)
{
    float a = fabsf(v);
    int n, c;
    if (a == 0) return 0;
    if (!(one > 0)) return -99999;
    n = (int)floor(pow((double)a/(double)one, 0.75) + 0.5);
    for (c = (n > 3 ? n - 3 : 0); c <= n + 3 && c <= 8206 + 4; c++)
    {
        if (L3_pow_43(c)*one == a)
            return v < 0 ? -c : c;
    }
    return -99999;  /* no exact match: should not happen */
}

static void fill_ix(hgr_t *g, const uint8_t *sfbtab, const float *scf)
{
    int i = 0, j = 0;
    while (i < 576 && sfbtab[j])
    {
        int w = sfbtab[j], k;
        for (k = 0; k < w && i < 576; k++, i++)
            g->ix[i] = ix_from_float(g->xr[i], scf[j]);
        j++;
    }
    for (; i < 576; i++) g->ix[i] = g->xr[i] == 0 ? 0 : -99999;
}

/* mirrors L3_decode_scalefactors up to L3_read_scalefactors, on a copy of the bitstream */
static void peek_iscf(const uint8_t *hdr, uint8_t *ist_pos_copy, const bs_t *bs, const L3_gr_info_t *gr, int ch, int32_t *out)
{
    static const uint8_t g_scf_partitions[3][28] = {
        { 6,5,5, 5,6,5,5,5,6,5, 7,3,11,10,0,0, 7, 7, 7,0, 6, 6,6,3, 8, 8,5,0 },
        { 8,9,6,12,6,9,9,9,6,9,12,6,15,18,0,0, 6,15,12,0, 6,12,9,6, 6,18,9,0 },
        { 9,9,6,12,9,9,9,9,9,9,12,6,18,18,0,0,12,12,12,0,12, 9,9,6,15,12,9,0 }
    };
    const uint8_t *scf_partition = g_scf_partitions[!!gr->n_short_sfb + !gr->n_long_sfb];
    uint8_t scf_size[4], iscf[40];
    int i, scfsi = gr->scfsi;
    bs_t b = *bs;
    if (HDR_TEST_MPEG1(hdr))
    {
        static const uint8_t g_scfc_decode[16] = { 0,1,2,3, 12,5,6,7, 9,10,11,13, 14,15,18,19 };
        int part = g_scfc_decode[gr->scalefac_compress];
        scf_size[1] = scf_size[0] = (uint8_t)(part >> 2);
        scf_size[3] = scf_size[2] = (uint8_t)(part & 3);
    } else
    {
        static const uint8_t g_mod[6*4] = { 5,5,4,4,5,5,4,1,4,3,1,1,5,6,6,1,4,4,4,1,4,3,1,1 };
        int k, modprod, sfc, ist = HDR_TEST_I_STEREO(hdr) && ch;
        sfc = gr->scalefac_compress >> ist;
        for (k = ist*3*4; sfc >= 0; sfc -= modprod, k += 4)
        {
            for (modprod = 1, i = 3; i >= 0; i--)
            {
                scf_size[i] = (uint8_t)(sfc / modprod % g_mod[k + i]);
                modprod *= g_mod[k + i];
            }
        }
        scf_partition += k;
        scfsi = -16;
    }
    memset(iscf, 0, sizeof(iscf));
    L3_read_scalefactors(iscf, ist_pos_copy, scf_size, scf_partition, &b, scfsi);
    for (i = 0; i < 40; i++) out[i] = iscf[i];
}

/*
 * Decode a whole buffer.  Returns number of frames (<= max_frames).
 * grs must hold max_frames*4 entries (index frame*4 + gr*2 + ch).
 * pcm (optional) receives interleaved output, max_samples total values.
 */
int h_decode_dump(const uint8_t *mp3, int mp3_bytes, hfr_t *frs, hgr_t *grs, int max_frames,
                  mp3d_sample_t *pcm, int max_samples, int *out_samples)
{
    static mp3dec_t dec;
    static mp3dec_scratch_t scratch;
    int pos = 0, nfr = 0, nsamp = 0;
    mp3dec_init(&dec);
    while (nfr < max_frames)
    {
        int i = 0, igr, frame_size = 0, success = 1, ch;
        const uint8_t *mp3p = mp3 + pos;
        int bytes = mp3_bytes - pos;
        const uint8_t *hdr;
        bs_t bs_frame[1];
        hfr_t *fr = frs + nfr;
        mp3d_sample_t *pcmp;
        int nch, ngr;

        if (bytes <= 4) break;
        if (dec.header[0] == 0xff && hdr_compare(dec.header, mp3p))
        {
            frame_size = hdr_frame_bytes(mp3p, dec.free_format_bytes) + hdr_padding(mp3p);
            if (frame_size != bytes && (frame_size + HDR_SIZE > bytes || !hdr_compare(mp3p, mp3p + frame_size)))
                frame_size = 0;
        }
        if (!frame_size)
        {
            memset(&dec, 0, sizeof(mp3dec_t));
            i = mp3d_find_frame(mp3p, bytes, &dec.free_format_bytes, &frame_size);
            if (!frame_size || i + frame_size > bytes)
                break;
        }
        hdr = mp3p + i;
        memcpy(dec.header, hdr, HDR_SIZE);
        memset(fr, 0, sizeof(*fr));
        fr->offset = pos + i;
        fr->frame_bytes = frame_size;
        fr->channels = nch = HDR_IS_MONO(hdr) ? 1 : 2;
        fr->hz = hdr_sample_rate_hz(hdr);
        fr->layer = 4 - HDR_GET_LAYER(hdr);
        fr->bitrate_kbps = hdr_bitrate_kbps(hdr);
        fr->mpeg1 = HDR_TEST_MPEG1(hdr) ? 1 : 0;
        fr->mode = HDR_GET_STEREO_MODE(hdr);
        fr->mode_ext = HDR_GET_STEREO_MODE_EXT(hdr);
        fr->sr_idx = HDR_GET_MY_SAMPLE_RATE(hdr);
        for (ch = 0; ch < 4; ch++) fr->hdr[ch] = hdr[ch];
        pos += i + frame_size;
        ngr = fr->mpeg1 ? 2 : 1;
        if (fr->layer != 3) { nfr++; continue; }

        bs_init(bs_frame, hdr + HDR_SIZE, frame_size - HDR_SIZE);
        if (HDR_IS_CRC(hdr)) get_bits(bs_frame, 16);
        {
            int main_data_begin = L3_read_side_info(bs_frame, scratch.gr_info, hdr);
            fr->main_data_begin = main_data_begin;
            if (main_data_begin < 0 || bs_frame->pos > bs_frame->limit)
            {
                mp3dec_init(&dec);
                fr->nsamples = 0;
                nfr++;
                continue;
            }
            success = L3_restore_reservoir(&dec, bs_frame, &scratch, main_data_begin);
            pcmp = pcm ? pcm + nsamp : NULL;
            if (success)
            {
                for (igr = 0; igr < ngr; igr++)
                {
                    L3_gr_info_t *gi = scratch.gr_info + igr*nch;
                    memset(scratch.grbuf[0], 0, 576*2*sizeof(float));
                    /* --- L3_decode with taps --- */
                    for (ch = 0; ch < nch; ch++)
                    {
                        hgr_t *g = grs + nfr*4 + igr*2 + ch;
                        L3_gr_info_t *gr = gi + ch;
                        int layer3gr_limit = scratch.bs.pos + gr->part_23_length;
                        uint8_t istcopy[39];
                        int k;
                        memcpy(istcopy, scratch.ist_pos[ch], 39);
                        g->part_23_length = gr->part_23_length; g->big_values = gr->big_values;
                        g->global_gain = gr->global_gain; g->scalefac_compress = gr->scalefac_compress;
                        g->block_type = gr->block_type; g->mixed_block_flag = gr->mixed_block_flag;
                        for (k = 0; k < 3; k++)
                        {
                            g->table_select[k] = gr->table_select[k]; g->region_count[k] = gr->region_count[k];
                            g->subblock_gain[k] = gr->subblock_gain[k];
                        }
                        g->preflag = gr->preflag; g->scalefac_scale = gr->scalefac_scale;
                        g->count1_table = gr->count1_table; g->scfsi = gr->scfsi;
                        g->n_long_sfb = gr->n_long_sfb; g->n_short_sfb = gr->n_short_sfb;
                        peek_iscf(dec.header, istcopy, &scratch.bs, gr, ch, g->iscf);
                        L3_decode_scalefactors(dec.header, scratch.ist_pos[ch], &scratch.bs, gr, scratch.scf, ch);
                        L3_huffman(scratch.grbuf[ch], &scratch.bs, gr, scratch.scf, layer3gr_limit);
                        memcpy(g->xr, scratch.grbuf[ch], sizeof(g->xr));
                        memcpy(g->scf, scratch.scf, sizeof(g->scf));
                        fill_ix(g, gr->sfbtab, scratch.scf);
                    }
                    if (HDR_TEST_I_STEREO(dec.header))
                        L3_intensity_stereo(scratch.grbuf[0], scratch.ist_pos[1], gi, dec.header);
                    else if (HDR_IS_MS_STEREO(dec.header))
                        L3_midside_stereo(scratch.grbuf[0], 576);
                    for (ch = 0; ch < nch; ch++)
                    {
                        L3_gr_info_t *gr = gi + ch;
                        int aa_bands = 31;
                        int n_long_bands = (gr->mixed_block_flag ? 2 : 0) << (int)(HDR_GET_MY_SAMPLE_RATE(dec.header) == 2);
                        if (gr->n_short_sfb)
                        {
                            aa_bands = n_long_bands - 1;
                            L3_reorder(scratch.grbuf[ch] + n_long_bands*18, scratch.syn[0], gr->sfbtab + gr->n_long_sfb);
                        }
                        L3_antialias(scratch.grbuf[ch], aa_bands);
                        L3_imdct_gr(scratch.grbuf[ch], dec.mdct_overlap[ch], gr->block_type, n_long_bands);
                        L3_change_sign(scratch.grbuf[ch]);
                    }
                    if (pcmp && nsamp + 576*nch*(igr + 1) <= max_samples)
                        mp3d_synth_granule(dec.qmf_state, scratch.grbuf[0], 18, nch, pcmp + 576*nch*igr, scratch.syn[0]);
                    else
                    {
                        static mp3d_sample_t junk[576*2];
                        mp3d_synth_granule(dec.qmf_state, scratch.grbuf[0], 18, nch, junk, scratch.syn[0]);
                    }
                }
                fr->nsamples = 576*ngr;
                nsamp += 576*ngr*nch;
            } else
                fr->nsamples = 0;
            L3_save_reservoir(&dec, &scratch);
        }
        nfr++;
    }
    if (out_samples) *out_samples = nsamp;
    return nfr;
}

/* --------------------------------------------------------------- synth -- */

/* minimal MSB-first bit writer used to build side info for L3_read_side_info */
typedef struct { uint8_t *p; int pos; } bw_t;
static void bw_put(bw_t *w, unsigned v, int n)
{
    while (n--)
    {
        if ((v >> n) & 1) w->p[w->pos >> 3] |= (uint8_t)(0x80 >> (w->pos & 7));
        w->pos++;
    }
}

/*
 * Fill L3_gr_info_t (sfbtab etc.) exactly as the decoder does, by packing a
 * side-info bitstream for one granule-set and parsing it with the decoder's
 * own L3_read_side_info.
 */
static int make_gr_info(const uint8_t *hdr, const hgr_t *g, int nch, L3_gr_info_t *out)
{
    uint8_t buf[64];
    bw_t w = { buf, 0 };
    bs_t bs;
    int mpeg1 = HDR_TEST_MPEG1(hdr) ? 1 : 0, ngr = mpeg1 ? 2 : 1, gr, ch;
    memset(buf, 0, sizeof(buf));
    if (mpeg1) { bw_put(&w, 0, 9); bw_put(&w, 0, nch == 1 ? 5 : 3); bw_put(&w, 0, 4*nch); }
    else { bw_put(&w, 0, 8); bw_put(&w, 0, nch == 1 ? 1 : 2); }
    for (gr = 0; gr < ngr; gr++)
        for (ch = 0; ch < nch; ch++)
        {
            const hgr_t *x = g + ch;   /* same params for both granules: we only use granule 0 */
            bw_put(&w, 0, 12);
            bw_put(&w, 0, 9);
            bw_put(&w, (unsigned)x->global_gain, 8);
            bw_put(&w, (unsigned)x->scalefac_compress, mpeg1 ? 4 : 9);
            if (x->block_type)
            {
                bw_put(&w, 1, 1);
                bw_put(&w, (unsigned)x->block_type, 2);
                bw_put(&w, (unsigned)x->mixed_block_flag, 1);
                bw_put(&w, 0, 10);
                bw_put(&w, (unsigned)x->subblock_gain[0], 3);
                bw_put(&w, (unsigned)x->subblock_gain[1], 3);
                bw_put(&w, (unsigned)x->subblock_gain[2], 3);
            } else
            {
                bw_put(&w, 0, 1);
                bw_put(&w, 0, 15);
                bw_put(&w, 0, 4);
                bw_put(&w, 0, 3);
            }
            if (mpeg1) bw_put(&w, (unsigned)x->preflag, 1);
            bw_put(&w, (unsigned)x->scalefac_scale, 1);
            bw_put(&w, 0, 1);
        }
    bs_init(&bs, buf, sizeof(buf));
    return L3_read_side_info(&bs, out, hdr) >= 0;
}

/* scale floats exactly as L3_decode_scalefactors computes them from iscf[] */
static void make_scf(const uint8_t *hdr, const L3_gr_info_t *gr, const int32_t *iscf_in, float *scf)
{
    int i, scf_shift = gr->scalefac_scale + 1;
    int iscf[40];
    float gain;
    for (i = 0; i < 40; i++) iscf[i] = (uint8_t)iscf_in[i];
    if (gr->n_short_sfb)
    {
        int sh = 3 - scf_shift;
        for (i = 0; i < gr->n_short_sfb; i += 3)
        {
            iscf[gr->n_long_sfb + i + 0] = (uint8_t)(iscf[gr->n_long_sfb + i + 0] + (gr->subblock_gain[0] << sh));
            iscf[gr->n_long_sfb + i + 1] = (uint8_t)(iscf[gr->n_long_sfb + i + 1] + (gr->subblock_gain[1] << sh));
            iscf[gr->n_long_sfb + i + 2] = (uint8_t)(iscf[gr->n_long_sfb + i + 2] + (gr->subblock_gain[2] << sh));
        }
    } else if (gr->preflag)
    {
        static const uint8_t g_preamp[10] = { 1,1,1,1,2,2,3,3,3,2 };
        for (i = 0; i < 10; i++)
            iscf[11 + i] = (uint8_t)(iscf[11 + i] + g_preamp[i]);
    }
    gain = scale_gain(gr->global_gain, HDR_IS_MS_STEREO(hdr));
    for (i = 0; i < (int)(gr->n_long_sfb + gr->n_short_sfb); i++)
        scf[i] = L3_ldexp_q2(gain, iscf[i] << scf_shift);
}

static void dequant(const L3_gr_info_t *gr, const float *scf, const int32_t *ix, float *dst)
{
    const uint8_t *sfb = gr->sfbtab;
    int i = 0, j = 0;
    memset(dst, 0, 576*sizeof(float));
    while (i < 576 && sfb[j])
    {
        int w = sfb[j], k;
        float one = scf[j];
        for (k = 0; k < w && i < 576; k++, i++)
        {
            int v = ix[i];
            if (v)
            {
                int a = v < 0 ? -v : v;
                float m = L3_pow_43(a)*one;
                dst[i] = v < 0 ? -m : m;
            }
        }
        j++;
    }
}

/*
 * The part of L3_decode after Huffman decoding + synthesis for one granule.
 * xr[ch*576] are dequantised spectra (bitstream order, before stereo).
 * sb (optional) receives the subband samples fed to the polyphase stage.
 */
static void post_huffman(mp3dec_t *dec, mp3dec_scratch_t *s, L3_gr_info_t *gi, int nch,
                         const float *xr, mp3d_sample_t *pcm, float *sb)
{
    int ch;
    memcpy(s->grbuf[0], xr, 576*nch*sizeof(float));
    if (nch == 1) memset(s->grbuf[1], 0, 576*sizeof(float));
    if (HDR_TEST_I_STEREO(dec->header))
        L3_intensity_stereo(s->grbuf[0], s->ist_pos[1], gi, dec->header);
    else if (HDR_IS_MS_STEREO(dec->header))
        L3_midside_stereo(s->grbuf[0], 576);
    for (ch = 0; ch < nch; ch++)
    {
        L3_gr_info_t *gr = gi + ch;
        int aa_bands = 31;
        int n_long_bands = (gr->mixed_block_flag ? 2 : 0) << (int)(HDR_GET_MY_SAMPLE_RATE(dec->header) == 2);
        if (gr->n_short_sfb)
        {
            aa_bands = n_long_bands - 1;
            L3_reorder(s->grbuf[ch] + n_long_bands*18, s->syn[0], gr->sfbtab + gr->n_long_sfb);
        }
        L3_antialias(s->grbuf[ch], aa_bands);
        L3_imdct_gr(s->grbuf[ch], dec->mdct_overlap[ch], gr->block_type, n_long_bands);
        L3_change_sign(s->grbuf[ch]);
    }
    if (sb) memcpy(sb, s->grbuf[0], 576*nch*sizeof(float));
    mp3d_synth_granule(dec->qmf_state, s->grbuf[0], 18, nch, pcm, s->syn[0]);
}

void h_dec_init(mp3dec_t *dec, const uint8_t *hdr4)
{
    memset(dec, 0, sizeof(*dec));
    memcpy(dec->header, hdr4, 4);
}

/*
 * Synthesise ngr granules from decoded parameters.
 * g: ngr*nch hgr_t (granule-major), hdrs: 4 bytes per granule (header
 * of the frame the granule belongs to; carries MPEG version, sample rate,
 * channel mode and MS/IS flags).  Uses fields: block_type, mixed_block_flag,
 * global_gain, scalefac_compress (for LSF sfb selection only), scalefac_scale,
 * preflag, subblock_gain, iscf, ix.  Also writes xr and scf back.
 * pcm: ngr*576*nch samples.  Returns 0 on success.
 */
int h_synth(mp3dec_t *dec, const uint8_t *hdrs, hgr_t *g, int ngr, int nch, mp3d_sample_t *pcm, float *sb)
{
    static mp3dec_scratch_t s;
    int igr, ch;
    for (igr = 0; igr < ngr; igr++)
    {
        L3_gr_info_t gi[4];
        float xr[2*576];
        hgr_t *gg = g + igr*nch;
        memcpy(dec->header, hdrs + 4*igr, 4);
        if (!make_gr_info(dec->header, gg, nch, gi)) return -1 - igr;
        for (ch = 0; ch < nch; ch++)
        {
            make_scf(dec->header, gi + ch, gg[ch].iscf, gg[ch].scf);
            dequant(gi + ch, gg[ch].scf, gg[ch].ix, xr + 576*ch);
            memcpy(gg[ch].xr, xr + 576*ch, sizeof(gg[ch].xr));
        }
        post_huffman(dec, &s, gi, nch, xr, pcm + 576*nch*igr, sb ? sb + 576*nch*igr : NULL);
    }
    return 0;
}

/*
 * Synthesise from spectra directly (system identification).  xr: ngr*nch*576,
 * block types per granule-channel in bt[], mixed flags in mx[].
 */
int h_synth_xr(mp3dec_t *dec, const uint8_t *hdrs, const float *xr, const int32_t *bt, const int32_t *mx,
               int ngr, int nch, mp3d_sample_t *pcm, float *sb)
{
    static mp3dec_scratch_t s;
    int igr, ch;
    for (igr = 0; igr < ngr; igr++)
    {
        L3_gr_info_t gi[4];
        hgr_t tmp[2];
        memset(tmp, 0, sizeof(tmp));
        memcpy(dec->header, hdrs + 4*igr, 4);
        for (ch = 0; ch < nch; ch++)
        {
            tmp[ch].block_type = bt[igr*nch + ch];
            tmp[ch].mixed_block_flag = mx[igr*nch + ch];
            tmp[ch].global_gain = 210;
        }
        if (!make_gr_info(dec->header, tmp, nch, gi)) return -1 - igr;
        post_huffman(dec, &s, gi, nch, xr + 576*nch*igr, pcm + 576*nch*igr, sb ? sb + 576*nch*igr : NULL);
    }
    return 0;
}

/* sfb table as the decoder uses it (widths, 0-terminated), for given block type */
int h_sfbtab(const uint8_t *hdr4, int block_type, int mixed, int32_t *widths, int32_t *n_long, int32_t *n_short)
{
    L3_gr_info_t gi[4];
    hgr_t tmp[2];
    int i;
    memset(tmp, 0, sizeof(tmp));
    tmp[0].block_type = tmp[1].block_type = block_type;
    tmp[0].mixed_block_flag = tmp[1].mixed_block_flag = mixed;
    if (!make_gr_info(hdr4, tmp, HDR_IS_MONO(hdr4) ? 1 : 2, gi)) return -1;
    for (i = 0; i < 40; i++) { widths[i] = gi[0].sfbtab[i]; if (!gi[0].sfbtab[i]) break; }
    *n_long = gi[0].n_long_sfb; *n_short = gi[0].n_short_sfb;
    return i;
}
