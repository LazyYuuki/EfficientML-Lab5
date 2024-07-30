#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

namespace matmul {
void MatmulOperator::mat_mul_loop_unrolling(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32
    float *scale = params->scales, *offset = params->offset;

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col += 4) {
            float acc0 = 0;
            float acc1 = 0;
            float acc2 = 0;
            float acc3 = 0;
            // Compute each block
            for (int ch = 0; ch < k;) {
                // pointer of the int8 activation
                const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
                // pointer of the int4 weights
                uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
                uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
                uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
                uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
                // scale of activation
                float s_a = params->A_scales[(row * k + ch) / block_size];
                // scale of weight
                float s_w0 = params->scales[(col * k + ch) / block_size];
                float s_w1 = params->scales[((col + 1) * k + ch) / block_size];
                float s_w2 = params->scales[((col + 2) * k + ch) / block_size];
                float s_w3 = params->scales[((col + 3) * k + ch) / block_size];
                // order of weights with QM_ARM:
                // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
                // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
                //               |--|
                //               4 bits
                //               |------|
                //               8 bits (byte)
                //            low|----------------------------------------------------------|high
                //               0                         128 bit                         127
                // process 16 bytes of weigths (128 bit) = 1 block for each of unrolled `col`
                // intermediate variable to store sum of integer multiplication and accumulation
                int intermediate_sum0 = 0, intermediate_sum1 = 0, intermediate_sum2 = 0, intermediate_sum3 = 0;
                for (int qj = 0; qj < 16; qj++) {
                    // TODO: decode a packed byte into two int8 in the range of (-8, 7)
                    uint8_t packed_int4_0 = w0_int4[qj];
                    signed char w0_de_0 = (packed_int4_0 & 0x0F) - 8.0;
                    signed char w0_de_16 = (packed_int4_0 >> 4) - 8.0;

                    uint8_t packed_int4_1 = w1_int4[qj];
                    signed char w1_de_0 = (packed_int4_1 & 0x0F) - 8.0;
                    signed char w1_de_16 = (packed_int4_1 >> 4) - 8.0;

                    uint8_t packed_int4_2 = w2_int4[qj];
                    signed char w2_de_0 = (packed_int4_2 & 0x0F) - 8.0;
                    signed char w2_de_16 = (packed_int4_2 >> 4) - 8.0;

                    uint8_t packed_int4_3 = w3_int4[qj];
                    signed char w3_de_0 = (packed_int4_3 & 0x0F) - 8.0;
                    signed char w3_de_16 = (packed_int4_3 >> 4) - 8.0;
                    // TODO: int8 multiply and accumulate operation
                    intermediate_sum0 += a_int8[qj] * w0_de_0;
                    intermediate_sum0 += a_int8[qj + 16] * w0_de_16;

                    intermediate_sum1 += a_int8[qj] * w1_de_0;
                    intermediate_sum1 += a_int8[qj + 16] * w1_de_16;

                    intermediate_sum2 += a_int8[qj] * w2_de_0;
                    intermediate_sum2 += a_int8[qj + 16] * w2_de_16;

                    intermediate_sum3 += a_int8[qj] * w3_de_0;
                    intermediate_sum3 += a_int8[qj + 16] * w3_de_16;
                }
                // dequantize the sum into floating point
                acc0 += (float)intermediate_sum0 * s_a * s_w0;
                acc1 += (float)intermediate_sum1 * s_a * s_w1;
                acc2 += (float)intermediate_sum2 * s_a * s_w2;
                acc3 += (float)intermediate_sum3 * s_a * s_w3;
                ch += block_size;
            }
            C->data_ptr[row * n + col] = acc0;
            C->data_ptr[row * n + col + 1] = acc1;
            C->data_ptr[row * n + col + 2] = acc2;
            C->data_ptr[row * n + col + 3] = acc3;
        }
    }
};
}  // namespace matmul
