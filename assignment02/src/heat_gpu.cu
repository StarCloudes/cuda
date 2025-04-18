// src/heat_gpu.cu

#include <iostream>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"

// TODO 1: Device kernel for heat propagation (each thread handles one element in the matrix)
__global__ void heat_propagate_kernel(float* next, const float* prev, int n, int m) {
    // TODO: Compute (row, col) from threadIdx + blockIdx
    // TODO: Handle boundary column (col == 0): just copy
    // TODO: Wrap-around indexing for jm2, jm1, jp1, jp2
    // TODO: Apply the 5-point stencil formula
}

// TODO 2: Device kernel to compute per-row averages (for -a flag)
__global__ void row_average_kernel(float* row_avg, const float* matrix, int n, int m) {
    // TODO: Each block handles one row, threads reduce over columns
    // TODO: Use shared memory or warp-level atomicAdd
}

// TODO 3: Host wrapper for heat propagation
void launch_heat_propagation(float* d_matrix_A, float* d_matrix_B, int n, int m, int iterations, cudaStream_t stream) {
    // TODO: Configure block/grid size
    // TODO: Launch propagation kernel alternating buffers
}

// TODO 4: Host wrapper for average computation
void launch_row_average(float* d_result_matrix, float* d_row_avg, int n, int m, cudaStream_t stream) {
    // TODO: Configure grid and block, launch row_average_kernel
}