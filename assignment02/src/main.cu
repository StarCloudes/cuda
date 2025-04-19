#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"
#include "heat_cpu.h"
#include <iomanip>
#include <chrono>

enum RunMode { MODE_BOTH, MODE_CPU_ONLY, MODE_GPU_ONLY };
RunMode mode = MODE_BOTH;


int main(int argc, char *argv[]) {
    int n = 32, m = 32, p = 10;
    bool cpu_only = false;
    bool do_avg = false;
    bool verbose = false;
    bool timing = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0) {
            do_avg = true;
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            mode = MODE_CPU_ONLY;
            timing = false;
        } else if (strcmp(argv[i], "-g") == 0) {
            mode = MODE_GPU_ONLY;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-t") == 0) {
            timing = true;
        } else {
            std::cerr << "Usage: ./heat_gpu [-n rows] [-m cols] [-p iters] [-a] [-g] [-v]\n";
            return 1;
        }
    }

    std::vector<float> hostA(n * m), hostB(n * m);

    // === Block size check ===
    const int blockSize = 4;
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

    cudaEvent_t ev_alloc_start, ev_alloc_stop;
    cudaEvent_t ev_copy_to_start, ev_copy_to_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_avg_start, ev_avg_stop;
    cudaEvent_t ev_copy_back_start, ev_copy_back_stop;
    float time_alloc = 0, time_copy_to = 0, time_kernel = 0, time_avg = 0, time_copy_back = 0;

    if (mode != MODE_CPU_ONLY) {
        cudaEventCreate(&ev_alloc_start); cudaEventCreate(&ev_alloc_stop);
        cudaEventRecord(ev_alloc_start);
        cudaMalloc(&d_A, n * m * sizeof(float));
        cudaMalloc(&d_B, n * m * sizeof(float));
        cudaEventRecord(ev_alloc_stop);
        cudaEventSynchronize(ev_alloc_stop);
        cudaEventElapsedTime(&time_alloc, ev_alloc_start, ev_alloc_stop);

        cudaEventCreate(&ev_copy_to_start); cudaEventCreate(&ev_copy_to_stop);
        cudaEventRecord(ev_copy_to_start);
        cudaMemcpy(d_A, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, hostA.data(), n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(ev_copy_to_stop);
        cudaEventSynchronize(ev_copy_to_stop);
        cudaEventElapsedTime(&time_copy_to, ev_copy_to_start, ev_copy_to_stop);
    }

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    if (timing && mode != MODE_CPU_ONLY) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    std::vector<float> cpuA = hostA;
    std::vector<float> cpuB = hostA;
    std::vector<float> cpu_avg(n);
    float cpu_time_ms = 0.0f;

    if (mode != MODE_GPU_ONLY) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_heat_propagation(cpuA, cpuB, n, m, p);
        auto cpu_mid = std::chrono::high_resolution_clock::now();
        compute_cpu_row_averages((p % 2 == 0 ? cpuA : cpuB), cpu_avg, n, m);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> duration_prop = cpu_mid - cpu_start;
        std::chrono::duration<float, std::milli> duration_avg = cpu_end - cpu_mid;
        cpu_time_ms = duration_prop.count() + duration_avg.count();
    }

    if (mode != MODE_CPU_ONLY) {
        cudaEventCreate(&ev_kernel_start); cudaEventCreate(&ev_kernel_stop);
        cudaEventRecord(ev_kernel_start);
        launch_heat_propagation(d_A, d_B, n, m, p);
        cudaEventRecord(ev_kernel_stop);
        cudaEventSynchronize(ev_kernel_stop);
        cudaEventElapsedTime(&time_kernel, ev_kernel_start, ev_kernel_stop);

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&gpuTime, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaEventCreate(&ev_copy_back_start); cudaEventCreate(&ev_copy_back_stop);
        cudaEventRecord(ev_copy_back_start);
        cudaMemcpy(hostB.data(), (p % 2 == 0 ? d_A : d_B), n * m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_copy_back_stop);
        cudaEventSynchronize(ev_copy_back_stop);
        cudaEventElapsedTime(&time_copy_back, ev_copy_back_start, ev_copy_back_stop);

        float total_sum = 0.0f, min_val = hostB[0], max_val = hostB[0];
        for (int i = 0; i < n * m; ++i) {
            total_sum += hostB[i];
            if (hostB[i] < min_val) min_val = hostB[i];
            if (hostB[i] > max_val) max_val = hostB[i];
        }
        std::cout << "GPU result: Final sum = " << total_sum
                  << ", min = " << min_val
                  << ", max = " << max_val << std::endl;
    }
    
    const std::vector<float>& output_matrix = cpu_only ? (p % 2 == 0 ? cpuA : cpuB) : hostB;

    std::cout << std::fixed << std::setprecision(6);
    if (verbose) {
        std::cout << "Final matrix after " << p << " iterations:\n";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                std::cout << std::setw(10) << output_matrix[i * m + j];
            }
            std::cout << "\n";
        }
    }


    if (timing) {
        std::cout << "\n[GPU Timing Breakdown]\n";
        std::cout << "GPU malloc time: " << time_alloc << " ms\n";
        std::cout << "GPU copy to device: " << time_copy_to << " ms\n";
        std::cout << "GPU kernel time: " << time_kernel << " ms\n";
        std::cout << "GPU row average time: " << time_avg << " ms\n";
        std::cout << "GPU copy back to host: " << time_copy_back << " ms\n";
        std::cout << "Total GPU compute time (kernel + avg): " << (time_kernel + time_avg) << " ms\n";
        std::cout << "Total GPU data transfer time: " << (time_alloc + time_copy_to + time_copy_back) << " ms\n";
        std::cout << "Total CPU compute time " << cpu_time_ms << " ms\n";
        std::cout << "Speedup (CPU / GPU kernel+avg): " << (cpu_time_ms / (time_kernel + time_avg)) << "x\n";
    }

    // if (!skip_cpu) {
    //     float max_diff = 0.0f;
    //     for (int i = 0; i < n * m; ++i) {
    //         float diff = std::fabs(output_matrix[i] - (p % 2 == 0 ? cpuA[i] : cpuB[i]));
    //         if (diff > 1e-4) {
    //             std::cout << "Mismatch at (" << i / m << "," << i % m << "): CPU=" 
    //                       << (p % 2 == 0 ? cpuA[i] : cpuB[i]) 
    //                       << " GPU=" << output_matrix[i] << " diff=" << diff << "\n";
    //         }
    //         if (diff > max_diff) max_diff = diff;
    //     }
    //     std::cout << "Max matrix difference: " << max_diff << "\n";
    // }

    if (do_avg) {
        std::cout << "Row averages after " << p << " iterations:\n";
        if (cpu_only) {
            for (int i = 0; i < n; ++i) {
                std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << cpu_avg[i] << "\n";
            }
        } else {
            cudaEventCreate(&ev_avg_start); cudaEventCreate(&ev_avg_stop);
            cudaEventRecord(ev_avg_start);
            float* d_avg;
            cudaMalloc(&d_avg, n * sizeof(float));
            launch_row_average((p % 2 == 0 ? d_A : d_B), d_avg, n, m);
            cudaEventRecord(ev_avg_stop);
            cudaEventSynchronize(ev_avg_stop);
            cudaEventElapsedTime(&time_avg, ev_avg_start, ev_avg_stop);

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