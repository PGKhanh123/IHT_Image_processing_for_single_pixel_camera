#include "iht.h"


// OPTIMIZED HARDWARE INSERTION NETWORK FOR FINDING THRESHOLD
data_t find_threshold(data_t x_tmp[N]) {
    #pragma HLS INLINE
    data_t top_k[K_SPARSITY];
    #pragma HLS ARRAY_PARTITION variable=top_k complete dim=1

    init_topk: for (int i = 0; i < K_SPARSITY; i++) {
        #pragma HLS UNROLL
        top_k[i] = 0;
    }

    find_topk: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1

        data_t val = (x_tmp[i] > 0) ? x_tmp[i] : (data_t)(-x_tmp[i]);
        data_t temp = val;

        compare_network: for (int j = 0; j < K_SPARSITY; j++) {
            #pragma HLS UNROLL
            if (temp > top_k[j]) {
                data_t swap = top_k[j];
                top_k[j] = temp;
                temp = swap;
            }
        }
    }
    return top_k[K_SPARSITY - 1];
}


// MAIN IHT CORE
void iht_core(data_t y[M], data_t A[M][N], data_t step_size, data_t x_hat[N]) {
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=step_size
    #pragma HLS INTERFACE ap_memory port=y
    #pragma HLS INTERFACE ap_memory port=A
    #pragma HLS INTERFACE ap_memory port=x_hat

    data_t local_y[M];
    data_t local_A[M][N];
    data_t local_x_hat[N];
    data_t local_x_tmp[N];
    data_t local_r[M];


    #pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=local_x_hat cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_x_tmp cyclic factor=8 dim=1

    // Data copy-in
    COPY_Y: for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        local_y[i] = y[i];
    }

    COPY_A_OUTER: for (int i = 0; i < M; i++) {
        COPY_A_INNER: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    INIT_X: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=8
        local_x_hat[j] = 0;
    }


    MAIN_ITERATION: for (int iter = 0; iter < MAX_ITER; iter++) {


        // Step 2: r = y - A * x (Using Hardware Adder Tree)

        CALC_RESIDUAL_OUTER: for (int i = 0; i < M; i++) {

            data_t sum_part[8];
            #pragma HLS ARRAY_PARTITION variable=sum_part complete dim=1

            init_sum: for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                sum_part[k] = 0;
            }

            CALC_RESIDUAL_INNER: for (int j = 0; j < N; j += 8) {
                #pragma HLS PIPELINE II=1
                res_mac: for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL
                    sum_part[k] += local_A[i][j + k] * local_x_hat[j + k];
                }
            }

            data_t total_sum = 0;
            res_reduction: for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                total_sum += sum_part[k];
            }

            local_r[i] = local_y[i] - total_sum;
        }


        // Step 3: Gradient Update (x_tmp = x + step * A^T * r)
        // Note: Row-major update. Parallel writes to distinct local_x_tmp indices
        INIT_TMP: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=8
            local_x_tmp[j] = local_x_hat[j];
        }

        CALC_GRADIENT: for (int i = 0; i < M; i++) {
            data_t r_step = local_r[i] * step_size;
            for (int j = 0; j < N; j += 8) {
                #pragma HLS PIPELINE II=1
                grad_mac: for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL
                    local_x_tmp[j + k] += local_A[i][j + k] * r_step;
                }
            }
        }


        // Step 4.1: Find Threshold
        data_t threshold = find_threshold(local_x_tmp);


        // Step 4.2: Hard Thresholding
        APPLY_THRESHOLD: for (int j = 0; j < N; j += 8) {
            #pragma HLS PIPELINE II=1
            ht_inner: for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                data_t val = local_x_tmp[j + k];
                data_t abs_val = (val > 0) ? val : (data_t)(-val);

                if (abs_val >= threshold) {
                    local_x_hat[j + k] = val;
                } else {
                    local_x_hat[j + k] = 0;
                }
            }
        }
    }

    // Data copy-out
    COPY_OUT: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        x_hat[j] = local_x_hat[j];
    }
}
