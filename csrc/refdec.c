/*
 * refdec: the reference decoder for this project.
 *
 * Pristine minimp3 (third_party/minimp3/minimp3.h, pinned commit), frame API
 * mp3dec_decode_frame() over the whole file, no gapless trimming, no
 * Xing/Info skipping.  Output: 16-bit PCM WAV (default build) or raw
 * little-endian float32 interleaved samples in [-1,1) (MINIMP3_FLOAT_OUTPUT
 * build, used only for analysis).
 *
 * usage: refdec in.mp3 out.wav|out.f32
 */
#define MINIMP3_IMPLEMENTATION
#include "../third_party/minimp3/minimp3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put32(FILE *f, unsigned v) { fputc(v & 255, f); fputc((v >> 8) & 255, f); fputc((v >> 16) & 255, f); fputc(v >> 24, f); }
static void put16(FILE *f, unsigned v) { fputc(v & 255, f); fputc((v >> 8) & 255, f); }

int main(int argc, char **argv)
{
    FILE *f;
    long len, pos = 0;
    uint8_t *buf;
    mp3dec_t dec;
    mp3dec_frame_info_t info;
    mp3d_sample_t pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
    mp3d_sample_t *out = NULL;
    size_t nout = 0, cap = 0;
    int channels = 0, hz = 0;

    if (argc < 3) { fprintf(stderr, "usage: %s in.mp3 out\n", argv[0]); return 2; }
    f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); len = ftell(f); fseek(f, 0, SEEK_SET);
    buf = (uint8_t *)malloc(len ? len : 1);
    if (fread(buf, 1, len, f) != (size_t)len) { fprintf(stderr, "read error\n"); return 1; }
    fclose(f);

    mp3dec_init(&dec);
    for (;;)
    {
        int samples = mp3dec_decode_frame(&dec, buf + pos, (int)(len - pos), pcm, &info);
        if (!info.frame_bytes)
            break;
        pos += info.frame_bytes;
        if (samples)
        {
            size_t n = (size_t)samples*info.channels;
            if (!channels) { channels = info.channels; hz = info.hz; }
            if (info.channels != channels) { fprintf(stderr, "channel count change unsupported\n"); return 1; }
            if (nout + n > cap) { cap = (nout + n)*2; out = (mp3d_sample_t *)realloc(out, cap*sizeof(*out)); }
            memcpy(out + nout, pcm, n*sizeof(*out));
            nout += n;
        }
    }

    f = fopen(argv[2], "wb");
    if (!f) { perror(argv[2]); return 1; }
#ifdef MINIMP3_FLOAT_OUTPUT
    fwrite(out, sizeof(float), nout, f);
#else
    {
        unsigned data = (unsigned)(nout*2);
        if (!channels) { channels = 1; hz = 44100; }
        fwrite("RIFF", 1, 4, f); put32(f, 36 + data); fwrite("WAVEfmt ", 1, 8, f);
        put32(f, 16); put16(f, 1); put16(f, channels); put32(f, hz);
        put32(f, hz*channels*2); put16(f, channels*2); put16(f, 16);
        fwrite("data", 1, 4, f); put32(f, data);
        fwrite(out, 2, nout, f);
    }
#endif
    fclose(f);
    fprintf(stderr, "%s: %d ch, %d Hz, %zu samples/ch\n", argv[1], channels, hz, channels ? nout/channels : 0);
    free(out); free(buf);
    return 0;
}
