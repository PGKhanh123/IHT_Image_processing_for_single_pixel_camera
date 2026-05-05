#ifndef IHT_H
#define IHT_H

#include <ap_fixed.h>

#define M 128
#define N 256
#define K_SPARSITY 40
#define MAX_ITER 100

typedef ap_fixed<24, 8, AP_RND , AP_WRAP> data_t;

void iht_core(data_t y[M], data_t A[M][N], data_t step_size, data_t x_hat[N]);

#endif
