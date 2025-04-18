#ifndef HEAT_GPU_CUH
#define HEAT_GPU_CUH

// Declare GPU kernel launchers

/**
 * @brief Launches the GPU heat propagation kernel for `p` iterations.
 * 
 * @param d_A Device pointer to matrix A (initial state)
 * @param d_B Device pointer to matrix B (workspace buffer)
 * @param n   Number of rows
 * @param m   Number of columns
 * @param p   Number of iterations
 */
void launch_heat_propagation(float* d_A, float* d_B, int n, int m, int p, cudaStream_t stream = 0);

/**
 * @brief Launches the GPU row average kernel.
 * 
 * @param d_result_matrix The final matrix after propagation
 * @param d_row_avg       Output buffer (device) for row averages
 * @param n               Number of rows
 * @param m               Number of columns
 * @param stream          CUDA stream (can be 0 for default)
 */
void launch_row_average(float* d_result_matrix, float* d_row_avg, int n, int m, cudaStream_t stream = 0);

#endif // HEAT_GPU_CUH