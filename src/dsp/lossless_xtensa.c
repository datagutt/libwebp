// Copyright 2026 Koios. All Rights Reserved.
//
// Xtensa PIE SIMD optimizations for VP8L (lossless) decoding on ESP32-S3.
//
// The VP8L hot path for our content is: Huffman decode (scalar, not
// vectorizable here) -> inverse transforms -> output conversion. This file
// vectorizes the two per-pixel passes that run over every decoded pixel:
//
//   - VP8LAddGreenToBlueAndRed: inverse of the subtract-green transform,
//     present in virtually every VP8L stream.
//   - VP8LConvertBGRAToRGBA: final canvas conversion for MODE_RGBA output.
//   - PredictorAdd 0/2/3/4/8/9: inverse of the spatial predictors that do
//     not depend on the just-decoded left pixel (black, top, top-right,
//     top-left and the two top-based averages).
//
// All process 4 pixels (16 bytes) per iteration in PIE Q registers.
//
// Safety: PIE shift/add lane semantics are hard to verify off-device, so
// VP8LDspInitXtensa() runs each vector function against the C reference on a
// test vector (covering unaligned heads, tails, sign/overflow byte patterns)
// and only installs it on an exact match. A mismatch logs once and keeps C.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_XTENSA_PIE)

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/dsp/xtensa_pie.h"

//------------------------------------------------------------------------------
// Broadcast constants for EE.VLDBC.32

static const uint32_t kMaskRedBlue = 0x00ff00ffu;     // R and B bytes of ARGB
static const uint32_t kMaskGreenAlpha = 0xff00ff00u;  // A and G bytes of ARGB
static const uint32_t kMaskLowByte = 0x000000ffu;
static const uint32_t kMaskByte2 = 0x00ff0000u;

//------------------------------------------------------------------------------
// Scalar single-pixel helpers (C reference semantics), used for unaligned
// heads and <4 pixel tails around the vector body.

static WEBP_INLINE uint32_t AddGreenOnePixel(uint32_t argb) {
    const uint32_t green = (argb >> 8) & 0xff;
    uint32_t red_blue = (argb & 0x00ff00ffu);
    red_blue += (green << 16) | green;
    red_blue &= 0x00ff00ffu;
    return (argb & 0xff00ff00u) | red_blue;
}

static WEBP_INLINE void BGRAToRGBAOnePixel(uint32_t argb, uint8_t* dst) {
    dst[0] = (argb >> 16) & 0xff;
    dst[1] = (argb >> 8) & 0xff;
    dst[2] = (argb >> 0) & 0xff;
    dst[3] = (argb >> 24) & 0xff;
}

//------------------------------------------------------------------------------
// AddGreenToBlueAndRed: dst[i] = argb with green added (mod 256) to R and B
//
// Per 32-bit lane:
//   green    = (v >> 8) & 0xff
//   red_blue = ((v & 0x00ff00ff) + (green | green << 16)) & 0x00ff00ff
//   out      = (v & 0xff00ff00) | red_blue
//
// The add runs as EE.VADDS.S16 on 16-bit lanes [B][R]: operands are at most
// 255 + 255 = 510, so signed saturation never triggers and it behaves as a
// plain add; the 0x00ff00ff mask afterwards provides the per-byte wraparound.

static void VP8LAddGreenToBlueAndRed_Xtensa(const uint32_t* src,
                                            int num_pixels, uint32_t* dst) {
    // The vector body needs src and dst to hit 16-byte alignment together
    if ((((uintptr_t)src ^ (uintptr_t)dst) & 15u) != 0) {
        VP8LAddGreenToBlueAndRed_C(src, num_pixels, dst);
        return;
    }

    // Scalar head until aligned
    while (num_pixels > 0 && ((uintptr_t)src & 15u) != 0) {
        *dst++ = AddGreenOnePixel(*src++);
        --num_pixels;
    }

    // Vector body: 4 pixels per iteration
    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_VLDBC_32(q6, &kMaskLowByte);
        PIE_VLDBC_32(q5, &kMaskGreenAlpha);

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, src);
            PIE_SET_SAR(8);
            PIE_VSR_32(q2, q0);         // v >> 8
            PIE_ANDQ(q2, q2, q6);       // green at bits [7:0]
            PIE_SET_SAR(16);
            PIE_VSL_32(q3, q2);         // green << 16
            PIE_ORQ(q2, q2, q3);        // green | green << 16
            PIE_ANDQ(q1, q0, q7);       // red_blue = v & 0x00ff00ff
            PIE_VADDS_S16(q1, q1, q2);  // [B+g][R+g], no saturation possible
            PIE_ANDQ(q1, q1, q7);       // per-byte wrap
            PIE_ANDQ(q0, q0, q5);       // keep A and G
            PIE_ORQ(q0, q0, q1);
            PIE_VST_128_IP(q0, dst);
        }
    }

    // Scalar tail
    while (num_pixels-- > 0) {
        *dst++ = AddGreenOnePixel(*src++);
    }
}

//------------------------------------------------------------------------------
// BGRA -> RGBA byte reorder (swap R and B within each 32-bit pixel)
//
// Per 32-bit lane: out = (v & 0xff00ff00) | ((v >> 16) & 0xff) | ((v << 16) & 0x00ff0000)

static void VP8LConvertBGRAToRGBA_Xtensa(const uint32_t* WEBP_RESTRICT src,
                                         int num_pixels,
                                         uint8_t* WEBP_RESTRICT dst) {
    if ((((uintptr_t)src ^ (uintptr_t)dst) & 15u) != 0) {
        VP8LConvertBGRAToRGBA_C(src, num_pixels, dst);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)src & 15u) != 0) {
        BGRAToRGBAOnePixel(*src++, dst);
        dst += 4;
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q5, &kMaskGreenAlpha);
        PIE_VLDBC_32(q6, &kMaskLowByte);
        PIE_VLDBC_32(q4, &kMaskByte2);
        PIE_SET_SAR(16);  // both shifts are by 16, set once

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, src);
            PIE_VSR_32(q1, q0);    // R to bits [7:0]
            PIE_ANDQ(q1, q1, q6);
            PIE_VSL_32(q2, q0);    // B to bits [23:16]
            PIE_ANDQ(q2, q2, q4);
            PIE_ANDQ(q0, q0, q5);  // keep A and G
            PIE_ORQ(q0, q0, q1);
            PIE_ORQ(q0, q0, q2);
            PIE_VST_128_IP(q0, dst);
        }
    }

    while (num_pixels-- > 0) {
        BGRAToRGBAOnePixel(*src++, dst);
        dst += 4;
    }
}

//------------------------------------------------------------------------------
// Predictor inverse (add) for the black and top predictors.
//
// out[i] = in[i] + pred (per byte, mod 256), with pred = 0xff000000 for
// predictor 0 and pred = upper[i] for predictor 2.
//
// PIE has no wrapping byte add and EE.VADDS.S32 saturates when the alpha
// bytes are large, so the add runs split across two EE.VADDS.S16 passes:
// even bytes [B][R] as (v & 0x00ff00ff) and odd bytes [G][A] as
// ((v >> 8) & 0x00ff00ff). Each 16-bit lane then holds one byte value
// (max sum 510), signed saturation never triggers, and the 0x00ff00ff mask
// afterwards provides the per-byte wraparound. SAR stays 8 for both the
// odd-byte extraction (VSR) and the recombine (VSL).

static const uint32_t kArgbBlack = 0xff000000u;
// Odd-byte plane of ARGB_BLACK: ((0xff000000 >> 8) & 0x00ff00ff)
static const uint32_t kBlackOddBytes = 0x00ff0000u;

static void PredictorAdd0_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    (void)upper;
    if ((((uintptr_t)in ^ (uintptr_t)out) & 15u) != 0) {
        VP8LPredictorsAdd_C[0](in, NULL, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, kArgbBlack);
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);    // even/odd byte-plane mask
        PIE_VLDBC_32(q6, &kBlackOddBytes);  // [G+0][A+0xff] addend
        PIE_SET_SAR(8);

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, in);
            PIE_ANDQ(q2, q0, q7);       // even bytes: += 0, pass through
            PIE_VSR_32(q4, q0);
            PIE_ANDQ(q4, q4, q7);       // odd bytes [G][A]
            PIE_VADDS_S16(q4, q4, q6);  // A += 0xff, no saturation possible
            PIE_ANDQ(q4, q4, q7);       // per-byte wrap
            PIE_VSL_32(q4, q4);         // back to G/A positions
            PIE_ORQ(q2, q2, q4);
            PIE_VST_128_IP(q2, out);
        }
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, kArgbBlack);
    }
}

static void PredictorAdd2_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    // upper is one canvas row above out, so for the row widths this firmware
    // decodes (multiples of 4 pixels) all three stay 16-byte congruent.
    if (((((uintptr_t)in ^ (uintptr_t)out) |
          ((uintptr_t)in ^ (uintptr_t)upper)) & 15u) != 0) {
        VP8LPredictorsAdd_C[2](in, upper, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, *upper++);
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_SET_SAR(8);

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, in);
            PIE_VLD_128_IP(q1, upper);
            PIE_ANDQ(q2, q0, q7);       // in even bytes [B][R]
            PIE_ANDQ(q3, q1, q7);       // upper even bytes
            PIE_VADDS_S16(q2, q2, q3);  // byte sums <= 510, no saturation
            PIE_ANDQ(q2, q2, q7);       // per-byte wrap
            PIE_VSR_32(q4, q0);
            PIE_ANDQ(q4, q4, q7);       // in odd bytes [G][A]
            PIE_VSR_32(q5, q1);
            PIE_ANDQ(q5, q5, q7);       // upper odd bytes
            PIE_VADDS_S16(q4, q4, q5);
            PIE_ANDQ(q4, q4, q7);
            PIE_VSL_32(q4, q4);         // back to G/A positions
            PIE_ORQ(q2, q2, q4);
            PIE_VST_128_IP(q2, out);
        }
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, *upper++);
    }
}

//------------------------------------------------------------------------------
// Predictor inverse (add) for the shifted-top predictors:
//   3: TR = upper[i+1]        4: TL = upper[i-1]
//   8: Average2(TL, T)        9: Average2(T, TR)
//
// The shifted upper vector is built from two aligned loads combined with
// EE.SRC.Q (SAR_BYTE = 4 for +1 pixel, 12 for -1 pixel). Its operand order
// is unverified off-device; the init self-check rejects a wrong guess and
// keeps C. The TL variants scalar-process four extra pixels before the
// vector body so the [i-4] block load never reads before memory the C
// reference would touch.
//
// Average2 runs per byte as ((a + b) >> 1): with even/odd bytes isolated in
// 16-bit lanes the sum is at most 510, so a 32-bit lane shift right by 1
// plus the 0x00ff00ff mask yields the exact per-byte average (the bit that
// crosses into the neighbor byte's field is masked off).

static WEBP_INLINE uint32_t Average2Pixel(uint32_t a0, uint32_t a1) {
    return (((a0 ^ a1) & 0xfefefefeu) >> 1) + (a0 & a1);
}

// Byte-wise q0 += q3 via even/odd split; result in q4. Clobbers q5, q6.
// Expects q7 = 0x00ff00ff broadcast and SAR = 8 on entry; leaves SAR = 8.
#define PIE_ADD_PIXELS_Q0_Q3_TO_Q4()  \
    do {                              \
        PIE_ANDQ(q4, q0, q7);         \
        PIE_ANDQ(q5, q3, q7);         \
        PIE_VADDS_S16(q4, q4, q5);    \
        PIE_ANDQ(q4, q4, q7);         \
        PIE_VSR_32(q5, q0);           \
        PIE_ANDQ(q5, q5, q7);         \
        PIE_VSR_32(q6, q3);           \
        PIE_ANDQ(q6, q6, q7);         \
        PIE_VADDS_S16(q5, q5, q6);    \
        PIE_ANDQ(q5, q5, q7);         \
        PIE_VSL_32(q5, q5);           \
        PIE_ORQ(q4, q4, q5);          \
    } while (0)

static void PredictorAdd3_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    if (((((uintptr_t)in ^ (uintptr_t)out) |
          ((uintptr_t)in ^ (uintptr_t)upper)) & 15u) != 0) {
        VP8LPredictorsAdd_C[3](in, upper, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, upper[1]);
        ++upper;
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_SET_SAR(8);
        PIE_SET_SAR_BYTE(4);

        while (n-- > 0) {
            PIE_VLD_128_IP(q1, upper);  // upper[i..i+3]; ptr now at i+4
            PIE_VLD_128(q2, upper);     // upper[i+4..i+7]
            PIE_SRC_Q(q3, q1, q2);      // TR = upper[i+1..i+4]
            PIE_VLD_128_IP(q0, in);
            PIE_ADD_PIXELS_Q0_Q3_TO_Q4();
            PIE_VST_128_IP(q4, out);
        }
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, upper[1]);
        ++upper;
    }
}

static void PredictorAdd4_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    if (((((uintptr_t)in ^ (uintptr_t)out) |
          ((uintptr_t)in ^ (uintptr_t)upper)) & 15u) != 0) {
        VP8LPredictorsAdd_C[4](in, upper, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, upper[-1]);
        ++upper;
        --num_pixels;
    }
    // TL guard: keep the vector body's upper[i-4] block inside C-touched
    // memory (see the section comment).
    int guard = num_pixels < 4 ? num_pixels : 4;
    num_pixels -= guard;
    while (guard-- > 0) {
        *out++ = VP8LAddPixels(*in++, upper[-1]);
        ++upper;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        const uint32_t* up_p = upper - 4;
        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_SET_SAR(8);
        PIE_SET_SAR_BYTE(12);

        while (n-- > 0) {
            PIE_VLD_128_IP(q1, up_p);  // upper[i-4..i-1]; ptr now at i
            PIE_VLD_128(q2, up_p);     // upper[i..i+3]
            PIE_SRC_Q(q3, q1, q2);     // TL = upper[i-1..i+2]
            PIE_VLD_128_IP(q0, in);
            PIE_ADD_PIXELS_Q0_Q3_TO_Q4();
            PIE_VST_128_IP(q4, out);
        }
        upper = up_p + 4;  // up_p started 4 pixels behind upper
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, upper[-1]);
        ++upper;
    }
}

// Shared vector body for the Average2 predictors. q1/q2 hold the two aligned
// upper blocks, q3 the SRC.Q-shifted block; the average of (qa, qb) is taken
// per byte, added to in, and stored. Clobbers q0, q4, q5, q6.
#define PIE_AVG_ADD_STORE(qa, qb)      \
    do {                               \
        /* even-byte average */        \
        PIE_ANDQ(q4, qa, q7);          \
        PIE_ANDQ(q5, qb, q7);          \
        PIE_VADDS_S16(q4, q4, q5);     \
        PIE_SET_SAR(1);                \
        PIE_VSR_32(q4, q4);            \
        PIE_ANDQ(q4, q4, q7);          \
        /* odd-byte average */         \
        PIE_SET_SAR(8);                \
        PIE_VSR_32(q5, qa);            \
        PIE_ANDQ(q5, q5, q7);          \
        PIE_VSR_32(q6, qb);            \
        PIE_ANDQ(q6, q6, q7);          \
        PIE_VADDS_S16(q5, q5, q6);     \
        PIE_SET_SAR(1);                \
        PIE_VSR_32(q5, q5);            \
        PIE_ANDQ(q5, q5, q7);          \
        /* in + average, per byte */   \
        PIE_SET_SAR(8);                \
        PIE_VLD_128_IP(q0, in);        \
        PIE_ANDQ(q6, q0, q7);          \
        PIE_VADDS_S16(q4, q6, q4);     \
        PIE_ANDQ(q4, q4, q7);          \
        PIE_VSR_32(q6, q0);            \
        PIE_ANDQ(q6, q6, q7);          \
        PIE_VADDS_S16(q5, q6, q5);     \
        PIE_ANDQ(q5, q5, q7);          \
        PIE_VSL_32(q5, q5);            \
        PIE_ORQ(q4, q4, q5);           \
        PIE_VST_128_IP(q4, out);       \
    } while (0)

static void PredictorAdd8_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    if (((((uintptr_t)in ^ (uintptr_t)out) |
          ((uintptr_t)in ^ (uintptr_t)upper)) & 15u) != 0) {
        VP8LPredictorsAdd_C[8](in, upper, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, Average2Pixel(upper[-1], upper[0]));
        ++upper;
        --num_pixels;
    }
    // TL guard, as in PredictorAdd4.
    int guard = num_pixels < 4 ? num_pixels : 4;
    num_pixels -= guard;
    while (guard-- > 0) {
        *out++ = VP8LAddPixels(*in++, Average2Pixel(upper[-1], upper[0]));
        ++upper;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        const uint32_t* up_p = upper - 4;
        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_SET_SAR(8);
        PIE_SET_SAR_BYTE(12);

        while (n-- > 0) {
            PIE_VLD_128_IP(q1, up_p);  // upper[i-4..i-1]
            PIE_VLD_128(q2, up_p);     // T  = upper[i..i+3]
            PIE_SRC_Q(q3, q1, q2);     // TL = upper[i-1..i+2]
            PIE_AVG_ADD_STORE(q3, q2);
        }
        upper = up_p + 4;  // up_p started 4 pixels behind upper
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, Average2Pixel(upper[-1], upper[0]));
        ++upper;
    }
}

static void PredictorAdd9_Xtensa(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
    if (((((uintptr_t)in ^ (uintptr_t)out) |
          ((uintptr_t)in ^ (uintptr_t)upper)) & 15u) != 0) {
        VP8LPredictorsAdd_C[9](in, upper, num_pixels, out);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)in & 15u) != 0) {
        *out++ = VP8LAddPixels(*in++, Average2Pixel(upper[0], upper[1]));
        ++upper;
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_SET_SAR(8);
        PIE_SET_SAR_BYTE(4);

        while (n-- > 0) {
            PIE_VLD_128_IP(q1, upper);  // T  = upper[i..i+3]
            PIE_VLD_128(q2, upper);     // upper[i+4..i+7]
            PIE_SRC_Q(q3, q1, q2);      // TR = upper[i+1..i+4]
            PIE_AVG_ADD_STORE(q1, q3);
        }
    }

    while (num_pixels-- > 0) {
        *out++ = VP8LAddPixels(*in++, Average2Pixel(upper[0], upper[1]));
        ++upper;
    }
}

//------------------------------------------------------------------------------
// Init-time self-check: run PIE and C implementations over byte patterns that
// exercise sign bits, carries and wraparound, plus unaligned heads and tails.
// Install the PIE version only on an exact output match.

#define PIE_CHECK_PIXELS 32

static void FillCheckInput(uint32_t* buf) {
    int i;
    static const uint32_t kPatterns[8] = {
        0x00000000u, 0xffffffffu, 0x80808080u, 0x7f7f7f7fu,
        0x01ff01ffu, 0xff01ff01u, 0xdeadbeefu, 0x00ff00ffu,
    };
    for (i = 0; i < PIE_CHECK_PIXELS; ++i) {
        // Mix fixed edge-case patterns with a ramp
        buf[i] = kPatterns[i & 7] ^ (uint32_t)(i * 0x01010101u);
    }
}

static int CheckAddGreen(void) {
    PIE_ALIGN static uint32_t in[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint32_t out_c[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint32_t out_pie[PIE_CHECK_PIXELS];
    FillCheckInput(in);

    // Aligned, vector body + tail (30 = head 0, body 28, tail 2)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LAddGreenToBlueAndRed_C(in, 30, out_c);
    VP8LAddGreenToBlueAndRed_Xtensa(in, 30, out_pie);
    if (memcmp(out_c, out_pie, sizeof(out_c)) != 0) return 0;

    // Unaligned head (src+1/dst+1 stay congruent mod 16)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LAddGreenToBlueAndRed_C(in + 1, 27, out_c + 1);
    VP8LAddGreenToBlueAndRed_Xtensa(in + 1, 27, out_pie + 1);
    return memcmp(out_c, out_pie, sizeof(out_c)) == 0;
}

// Both predictor checks share this driver: the C reference comes from
// VP8LPredictorsAdd_C, which VP8LDspInit populates before calling the
// per-arch init functions.
static int CheckPredictorAdd(int pred, VP8LPredictorAddSubFunc pie_func) {
    PIE_ALIGN static uint32_t in[PIE_CHECK_PIXELS];
    // The shifted-top predictors read upper[-1] (scalar TL) and their block
    // loads reach a few pixels past the span; pad both sides. `up` stays at
    // a 16-byte offset so pointer congruence (and thus the vector paths)
    // is preserved.
    PIE_ALIGN static uint32_t up_store[PIE_CHECK_PIXELS + 12];
    PIE_ALIGN static uint32_t out_c[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint32_t out_pie[PIE_CHECK_PIXELS];
    uint32_t* const up = up_store + 4;
    int i;
    FillCheckInput(in);
    for (i = 0; i < (int)(sizeof(up_store) / sizeof(up_store[0])); ++i) {
        up_store[i] = in[(PIE_CHECK_PIXELS - 1 - i) & (PIE_CHECK_PIXELS - 1)] ^
                      (0xa5a5a5a5u + (uint32_t)i * 0x01010101u);
    }

    // Aligned, vector body + tail (30 = head 0, body 28, tail 2)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LPredictorsAdd_C[pred](in, up, 30, out_c);
    pie_func(in, up, 30, out_pie);
    if (memcmp(out_c, out_pie, sizeof(out_c)) != 0) return 0;

    // Unaligned head (all pointers stay congruent mod 16)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LPredictorsAdd_C[pred](in + 1, up + 1, 27, out_c + 1);
    pie_func(in + 1, up + 1, 27, out_pie + 1);
    return memcmp(out_c, out_pie, sizeof(out_c)) == 0;
}

static int CheckConvertBGRAToRGBA(void) {
    PIE_ALIGN static uint32_t in[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint8_t out_c[PIE_CHECK_PIXELS * 4];
    PIE_ALIGN static uint8_t out_pie[PIE_CHECK_PIXELS * 4];
    FillCheckInput(in);

    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LConvertBGRAToRGBA_C(in, 30, out_c);
    VP8LConvertBGRAToRGBA_Xtensa(in, 30, out_pie);
    if (memcmp(out_c, out_pie, sizeof(out_c)) != 0) return 0;

    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LConvertBGRAToRGBA_C(in + 1, 27, out_c + 4);
    VP8LConvertBGRAToRGBA_Xtensa(in + 1, 27, out_pie + 4);
    return memcmp(out_c, out_pie, sizeof(out_c)) == 0;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LDspInitXtensa(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitXtensa(void) {
    if (CheckAddGreen()) {
        VP8LAddGreenToBlueAndRed = VP8LAddGreenToBlueAndRed_Xtensa;
    } else {
        printf("libwebp: PIE AddGreenToBlueAndRed self-check failed, using C\n");
    }
    if (CheckConvertBGRAToRGBA()) {
        VP8LConvertBGRAToRGBA = VP8LConvertBGRAToRGBA_Xtensa;
    } else {
        printf("libwebp: PIE ConvertBGRAToRGBA self-check failed, using C\n");
    }
    if (CheckPredictorAdd(0, PredictorAdd0_Xtensa)) {
        VP8LPredictorsAdd[0] = PredictorAdd0_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd0 self-check failed, using C\n");
    }
    if (CheckPredictorAdd(2, PredictorAdd2_Xtensa)) {
        VP8LPredictorsAdd[2] = PredictorAdd2_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd2 self-check failed, using C\n");
    }
    if (CheckPredictorAdd(3, PredictorAdd3_Xtensa)) {
        VP8LPredictorsAdd[3] = PredictorAdd3_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd3 self-check failed, using C\n");
    }
    if (CheckPredictorAdd(4, PredictorAdd4_Xtensa)) {
        VP8LPredictorsAdd[4] = PredictorAdd4_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd4 self-check failed, using C\n");
    }
    if (CheckPredictorAdd(8, PredictorAdd8_Xtensa)) {
        VP8LPredictorsAdd[8] = PredictorAdd8_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd8 self-check failed, using C\n");
    }
    if (CheckPredictorAdd(9, PredictorAdd9_Xtensa)) {
        VP8LPredictorsAdd[9] = PredictorAdd9_Xtensa;
    } else {
        printf("libwebp: PIE PredictorAdd9 self-check failed, using C\n");
    }
}

#else  // !WEBP_USE_XTENSA_PIE

WEBP_DSP_INIT_STUB(VP8LDspInitXtensa)

#endif  // WEBP_USE_XTENSA_PIE
