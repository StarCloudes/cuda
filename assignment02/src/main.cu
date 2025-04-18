#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"
#include "heat_cpu.h"
#include <iomanip>
#include <chrono>

int main(int argc, char *argv[]) {
    int n = 32, m = 32, p = 10;
    bool do_avg = false;
    bool skip_cpu = false;
    bool verbose = false;
    bool timing = false;
    bool cpu_only = false;
    
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
        } else if (strcmp(argv[i], "-t") == 0) {
            timing = true;
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            cpu_only = true;
            skip_cpu = false;
            do_avg = true;
            timing = false;
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

    if (!cpu_only) {
        cudaMalloc(&d_A, n * m * sizeof(float));
        cudaMalloc(&d_B, n * m * sizeof(float));

        cudaMemcpy(d_A, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    if (timing && !cpu_only) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    std::vector<float> cpuA = hostA;
    std::vector<float> cpuB = hostA;
    std::vector<float> cpu_avg(n);
    float cpu_time_ms = 0.0f;

    if (!skip_cpu) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_heat_propagation(cpuA, cpuB, n, m, p);
        compute_cpu_row_averages((p % 2 == 0 ? cpuA : cpuB), cpu_avg, n, m);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = cpu_end - cpu_start;
        cpu_time_ms = duration.count();
    }

    if (!cpu_only) {
        launch_heat_propagation(d_A, d_B, n, m, p);

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpuTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            std::cout << "GPU propagation time: " << gpuTime << " ms\n";
        }

        cudaMemcpy(hostB.data(), (p % 2 == 0 ? d_A : d_B), n * m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    const std::vector<float>& output_matrix = cpu_only ? (p % 2 == 0 ? cpuA : cpuB) : hostB;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Final matrix after " << p << " iterations:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << std::setw(10) << output_matrix[i * m + j];
        }
        std::cout << "\n";
    }

    if (timing && !skip_cpu) {
        std::cout << "CPU propagation + average time: " << cpu_time_ms << " ms\n";
        std::cout << "Speedup (CPU/GPU): " << (cpu_time_ms / gpuTime) << "x\n";
    }

    if (!skip_cpu) {
        float max_diff = 0.0f;
        for (int i = 0; i < n * m; ++i) {
            float diff = std::fabs(output_matrix[i] - (p % 2 == 0 ? cpuA[i] : cpuB[i]));
            if (diff > 1e-4) {
                std::cout << "Mismatch at (" << i / m << "," << i % m << "): CPU=" 
                          << (p % 2 == 0 ? cpuA[i] : cpuB[i]) 
                          << " GPU=" << output_matrix[i] << " diff=" << diff << "\n";
            }
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << "Max matrix difference: " << max_diff << "\n";
    }

    if (do_avg) {
        std::cout << "Row averages after " << p << " iterations:\n";
        if (cpu_only) {
            for (int i = 0; i < n; ++i) {
                std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << cpu_avg[i] << "\n";
            }
        } else {
            float* d_avg;
            cudaMalloc(&d_avg, n * sizeof(float));
            launch_row_average((p % 2 == 0 ? d_A : d_B), d_avg, n, m);

            std::vector<float> host_avg(n);
            cudaMemcpy(host_avg.data(), d_avg, n * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < n; ++i) {
                std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << host_avg[i] << "\n";
            }
            cudaFree(d_avg);
        }
    }

    if (!cpu_only) {
        cudaFree(d_A);
        cudaFree(d_B);
    }
    
    return 0;
}