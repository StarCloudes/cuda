#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"

int main() {
    const int n = 5, m = 5, p = 10;

    std::vector<float> hostA(n * m), hostB(n * m);

    
    for (int i = 0; i < n; ++i) {
        float boundary = 0.98f * (i + 1) * (i + 1) / float(n * n);
        hostA[i * m + 0] = boundary;
        for (int j = 1; j < m; ++j) {
            float factor = float((m - j) * (m - j)) / float(m * m);
            hostA[i * m + j] = boundary * factor;
        }
    }

    float *d_A, *d_B;
    cudaMalloc(&d_A, n * m * sizeof(float));
    cudaMalloc(&d_B, n * m * sizeof(float));

    cudaMemcpy(d_A, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);

    launch_heat_propagation(d_A, d_B, n, m, p);

    cudaMemcpy(hostB.data(), (p % 2 == 0 ? d_A : d_B), n * m * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final matrix after " << p << " iterations:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << hostB[i * m + j] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}