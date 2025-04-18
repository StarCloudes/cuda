#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"
#include <iomanip>

int main(int argc, char *argv[]) {
    int n = 32, m = 32, p = 10;
    bool do_avg = false;
    bool skip_cpu = false;
    bool verbose = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0) {
            do_avg = true;
        } else if (strcmp(argv[i], "-c") == 0) {
            skip_cpu = true;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else {
            std::cerr << "Usage: ./heat_gpu [-n rows] [-m cols] [-p iters] [-a] [-c] [-v]\n";
            return 1;
        }
    }

    std::vector<float> hostA(n * m), hostB(n * m);

    // === Block size check ===
    const int blockSize = 256;
    int totalSize = n * m;
    if (totalSize % blockSize != 0) {
        std::cerr << "ERROR: Block size (" << blockSize
                << ") must divide matrix size (n*m = "
                << totalSize << ") evenly.\n";
        return 1;
    }
    
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
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Final matrix after " << p << " iterations:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << std::setw(10) << hostB[i * m + j];
        }
        std::cout << "\n";
    }

    if (do_avg) {
        float* d_avg;
        cudaMalloc(&d_avg, n * sizeof(float));
        launch_row_average((p % 2 == 0 ? d_A : d_B), d_avg, n, m);
        
        std::vector<float> host_avg(n);
        cudaMemcpy(host_avg.data(), d_avg, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "Row averages after " << p << " iterations:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << host_avg[i] << "\n";
        }
        cudaFree(d_avg);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}