#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            // order of weights with QM_ARM:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
            // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         128 bit                         127
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            // pointer of the int4 weights
            const unsigned char *w_start = &B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const signed char *a_start = &A->int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];

            const int num_block = k / block_size;
            // Compute each block
            for (int q = 0; q < num_block; q++) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);
                w_start += 16;

                /*
                   We will accelerate the program using ARM Intrinsics. You can check the documentation of operations
                   at: https://developer.arm.com/architectures/instruction-sets/intrinsics
                */
                // TODO: decode the lower and upper half of the weights as int8x16_t
                // Hint:
                // (1) use `vandq_u8` with the mask_low4bit to get the lower half
                // (2) use `vshrq_n_u8` to right shift 4 bits and get the upper half
                // (3) use `vreinterpretq_s8_u8` to interpret the  vector as int8
                // lowbit mask
                const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);

                const uint8x16_t lower_half = vandq_u8(w0, mask_low4bit);
                const uint8x16_t upper_half = vshrq_n_u8(w0, 4);

                const int8x16_t signed_lower_half = vreinterpretq_s8_u8(lower_half);
                const int8x16_t signed_upper_half = vreinterpretq_s8_u8(upper_half);

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `vsubq_s8` to the lower-half and upper-half vectors of weights
                const int8x16_t offsets = vdupq_n_s8(8);
                const int8x16_t w_0 = vsubq_s8(signed_lower_half, offsets);
                const int8x16_t w_64 = vsubq_s8(signed_upper_half, offsets);

                // load 32 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                a_start += 32;

                // TODO: perform dot product and store the result into the intermediate sum, int_sum0
                // Hint: use `vdotq_s32` to compute sumv0 = a0 * lower-half weights + a1 * upper-half weights
                // int32x4 vector to store intermediate sum
                int32x4_t int_sum0;
                int_sum0 = vdupq_n_s32(0);
                int_sum0 = vdotq_s32(int_sum0, w_0, a0);
                int_sum0 = vdotq_s32(int_sum0, w_64, a1);

                float s_0 = *s_a++ * *s_w++;
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
            }
            C->data_ptr[row * n + col] = vaddvq_f32(sumv0);
        }
    }
};
}  // namespace matmul
