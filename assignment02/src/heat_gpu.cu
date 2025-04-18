// src/heat_gpu.cu

#include <iostream>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"

// TODO 1: Device kernel for heat propagation (each thread handles one element in the matrix)
__global__ void heat_propagate_kernel(float* next, const float* prev, int n, int m) {
    // TODO: Compute (row, col) from threadIdx + blockIdx
    int row = blockIdx.y * blockDim.y + threadIdx.y; // which row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // which column

    if (row >= n || col >= m) return;
    int idx = row * m + col;

    // TODO: Handle boundary column (col == 0): just copy
    if (col == 0) {
        // Boundary column: copy directly
        next[idx] = prev[idx];
        return;
    }

    // TODO: Wrap-around indexing for jm2, jm1, jp1, jp2
    // wrap-around indexing
    int jm2 = (col - 2 + m) % m;
    int jm1 = (col - 1 + m) % m;
    int jp1 = (col + 1) % m;
    int jp2 = (col + 2) % m;

    // TODO: Apply the 5-point stencil formula
    // Convert to linear indices
    int i_jm2 = row * m + jm2;
    int i_jm1 = row * m + jm1;
    int i_j   = row * m + col;
    int i_jp1 = row * m + jp1;
    int i_jp2 = row * m + jp2;

    float sum = 1.60f * prev[i_jm2]
              + 1.55f * prev[i_jm1]
              +         prev[i_j]
              + 0.60f * prev[i_jp1]
              + 0.25f * prev[i_jp2];
    next[idx] = sum / 5.0f;

}

// TODO 2: Device kernel to compute per-row averages (for -a flag)
__global__ void row_average_kernel(float* row_avg, const float* matrix, int n, int m) {
    // TODO: Each block handles one row, threads reduce over columns
    // TODO: Use shared memory or warp-level atomicAdd
}

// TODO 3: Host wrapper for heat propagation
void launch_heat_propagation(float* d_matrix_A, float* d_matrix_B, int n, int m, int iterations, cudaStream_t stream) {
    // TODO: Configure block/grid size
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    // TODO: Launch propagation kernel alternating buffers             
    for (int iter = 0; iter < p; ++iter) {
        if (iter % 2 == 0) {
            heat_propagate_kernel<<<gridDim, blockDim>>>(d_B, d_A, n, m);
        } else {
            heat_propagate_kernel<<<gridDim, blockDim>>>(d_A, d_B, n, m);
        }
    }
    cudaDeviceSynchronize();
}

// TODO 4: Host wrapper for average computation
void launch_row_average(float* d_result_matrix, float* d_row_avg, int n, int m, cudaStream_t stream) {
    // TODO: Configure grid and block, launch row_average_kernel
}