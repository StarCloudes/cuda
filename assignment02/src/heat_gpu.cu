// src/heat_gpu.cu

#include <iostream>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"
#include "real.h"

// TODO 1: Device kernel for heat propagation (each thread handles one element in the matrix)
__global__ void heat_propagate_kernel(real_t* next, const real_t* prev, int n, int m) {
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

    real_t sum = 1.60f * prev[i_jm2]
              + 1.55f * prev[i_jm1]
              +         prev[i_j]
              + 0.60f * prev[i_jp1]
              + 0.25f * prev[i_jp2];
    next[idx] = sum / 5.0f;

}

// TODO 2: Device kernel to compute per-row averages (for -a flag)
__global__ void row_average_kernel(real_t* row_avg, const real_t* matrix, int n, int m) {
    // TODO: Each block handles one row, threads reduce over columns
    // Each block is responsible for processing one row 
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ real_t partial_sum[256]; // Max threads per block

    real_t local = 0.0f;

    for (int j = tid; j < m; j += blockDim.x) {
        local += matrix[row * m + j];
    }
    // TODO: Use shared memory or warp-level atomicAdd
    partial_sum[tid] = local;
    __syncthreads();

    // Row Average Calculation
    if (tid == 0) {
        real_t sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += partial_sum[i];
        }
        row_avg[row] = sum / m;
    }
}

// TODO 3: Host wrapper for heat propagation
void launch_heat_propagation(real_t* d_matrix_A, real_t* d_matrix_B, int n, int m, int iterations, cudaStream_t stream) {
    // TODO: Configure block/grid size
    //dim3 blockDim(16, 16);
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    // TODO: Launch propagation kernel alternating buffers             
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            heat_propagate_kernel<<<gridDim, blockDim>>>(d_matrix_B, d_matrix_A, n, m);
        } else {
            heat_propagate_kernel<<<gridDim, blockDim>>>(d_matrix_A, d_matrix_B, n, m);
        }
    }
    cudaDeviceSynchronize();
}

// TODO 4: Host wrapper for average computation
void launch_row_average(real_t* d_result_matrix, real_t* d_row_avg, int n, int m, cudaStream_t stream) {
    // TODO: Configure grid and block, launch row_average_kernel
    int threads = 256; // adjust if m is very small
    row_average_kernel<<<n, threads, 0, stream>>>(d_row_avg, d_result_matrix, n, m);
    cudaDeviceSynchronize();
}