#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "heat_gpu.cuh"
#include "heat_cpu.h"
#include <iomanip>
#include <chrono>
#include "real.h"

// Enum to control execution mode: both CPU and GPU, or only CPU, or only GPU
enum RunMode { MODE_BOTH, MODE_CPU_ONLY, MODE_GPU_ONLY };
RunMode mode = MODE_BOTH;

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    int n = 32, m = 32, p = 10;
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

    // Allocate host matrices for input and output
    std::vector<real_t> hostA(n * m), hostB(n * m);

    // === Block size check ===
    const int blockSize = 4;
    int totalSize = n * m;
    if (totalSize % blockSize != 0) {
        std::cerr << "ERROR: Block size (" << blockSize
                << ") must divide matrix size (n*m = "
                << totalSize << ") evenly.\n";
        return 1;
    }
    
    // Initialize hostA with boundary and initial conditions
    for (int i = 0; i < n; ++i) {
        real_t boundary = 0.98f * (i + 1) * (i + 1) / real_t(n * n);
        hostA[i * m + 0] = boundary;
        for (int j = 1; j < m; ++j) {
            real_t factor = real_t((m - j) * (m - j)) / real_t(m * m);
            hostA[i * m + j] = boundary * factor;
        }
    }

    // Device pointers for GPU matrices
    real_t *d_A, *d_B;

    // GPU memory allocation and data transfer to device
    cudaEvent_t ev_alloc_start, ev_alloc_stop;
    cudaEvent_t ev_copy_to_start, ev_copy_to_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_avg_start, ev_avg_stop;
    cudaEvent_t ev_copy_back_start, ev_copy_back_stop;
    float time_alloc = 0, time_copy_to = 0, time_kernel = 0, time_avg = 0, time_copy_back = 0;

    if (mode != MODE_CPU_ONLY) {
        cudaEventCreate(&ev_alloc_start); cudaEventCreate(&ev_alloc_stop);
        cudaEventRecord(ev_alloc_start);
        cudaMalloc(&d_A, n * m * sizeof(real_t));
        cudaMalloc(&d_B, n * m * sizeof(real_t));
        cudaEventRecord(ev_alloc_stop);
        cudaEventSynchronize(ev_alloc_stop);
        cudaEventElapsedTime(&time_alloc, ev_alloc_start, ev_alloc_stop);

        cudaEventCreate(&ev_copy_to_start); cudaEventCreate(&ev_copy_to_stop);
        cudaEventRecord(ev_copy_to_start);
        cudaMemcpy(d_A, hostA.data(), n * m * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, hostA.data(), n * m * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaEventRecord(ev_copy_to_stop);
        cudaEventSynchronize(ev_copy_to_stop);
        cudaEventElapsedTime(&time_copy_to, ev_copy_to_start, ev_copy_to_stop);
    }

    // Start overall GPU timer if requested
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    if (timing && mode != MODE_CPU_ONLY) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    std::vector<real_t> cpuA = hostA;
    std::vector<real_t> cpuB = hostA;
    std::vector<real_t> cpu_avg(n);
    float cpu_time_ms = 0.0f;

    // Run CPU version of the heat propagation
    if (mode != MODE_GPU_ONLY) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_heat_propagation(cpuA, cpuB, n, m, p);
        auto cpu_mid = std::chrono::high_resolution_clock::now();
        compute_cpu_row_averages((p % 2 == 0 ? cpuA : cpuB), cpu_avg, n, m);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<real_t, std::milli> duration_prop = cpu_mid - cpu_start;
        std::chrono::duration<real_t, std::milli> duration_avg = cpu_end - cpu_mid;
        cpu_time_ms = duration_prop.count() + duration_avg.count();
    }

    // Run GPU version of the heat propagation
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
        cudaMemcpy(hostB.data(), (p % 2 == 0 ? d_A : d_B), n * m * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_copy_back_stop);
        cudaEventSynchronize(ev_copy_back_stop);
        cudaEventElapsedTime(&time_copy_back, ev_copy_back_start, ev_copy_back_stop);

        real_t total_sum = 0.0f, min_val = hostB[0], max_val = hostB[0];
        for (int i = 0; i < n * m; ++i) {
            total_sum += hostB[i];
            if (hostB[i] < min_val) min_val = hostB[i];
            if (hostB[i] > max_val) max_val = hostB[i];
        }
        std::cout << "GPU result: Final sum = " << total_sum
                  << ", min = " << min_val
                  << ", max = " << max_val << std::endl;
    }
    
    // Choose output matrix based on execution mode
    const std::vector<real_t>& output_matrix = mode == MODE_CPU_ONLY ? (p % 2 == 0 ? cpuA : cpuB) : hostB;

    // Print final matrix if verbose flag is set
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

    // Print timing breakdown if timing flag is set
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

    // Compute and print row averages (CPU or GPU)
    if (do_avg) {
        std::cout << "Row averages after " << p << " iterations:\n";
        if (mode == MODE_CPU_ONLY) {
            for (int i = 0; i < n; ++i) {
                std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << cpu_avg[i] << "\n";
            }
        } 
        else if(mode == MODE_BOTH) {
            cudaEventCreate(&ev_avg_start); cudaEventCreate(&ev_avg_stop);
            cudaEventRecord(ev_avg_start);
            real_t* d_avg;
            cudaMalloc(&d_avg, n * sizeof(real_t));
            launch_row_average((p % 2 == 0 ? d_A : d_B), d_avg, n, m);
            cudaEventRecord(ev_avg_stop);
            cudaEventSynchronize(ev_avg_stop);
            cudaEventElapsedTime(&time_avg, ev_avg_start, ev_avg_stop);

            std::vector<real_t> host_avg(n);
            cudaMemcpy(host_avg.data(), d_avg, n * sizeof(real_t), cudaMemcpyDeviceToHost);

            for (int i = 0; i < n; ++i) {
                std::cout << "Row " << std::setw(3) << i << " avg = " << std::setw(10) << host_avg[i] << "\n";
            }

            real_t max_avg_diff = 0.0f;
            for (int i = 0; i < n; ++i) {
                real_t diff = std::fabs(cpu_avg[i] - host_avg[i]);
                if (diff > 1e-6f) {
                    std::cout << "Row avg mismatch at row " << i << ": CPU=" << cpu_avg[i]
                              << ", GPU=" << host_avg[i]
                              << ", diff=" << diff << "\n";
                }
                if (diff > max_avg_diff) max_avg_diff = diff;
            }
            std::cout << "Max row average difference: " << max_avg_diff << "\n";

            cudaFree(d_avg);
        }
    }

    // Compare CPU and GPU matrices element-wise
    if (mode == MODE_BOTH) {
        const std::vector<real_t>& cpu_matrix = (p % 2 == 0 ? cpuA : cpuB);
        real_t max_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int idx = i * m + j;
                real_t diff = std::fabs(cpu_matrix[idx] - hostB[idx]);
                if (diff > 1e-6f) {
                    std::cout << "Mismatch at (" << i << "," << j << "): "
                              << "CPU=" << cpu_matrix[idx]
                              << ", GPU=" << hostB[idx]
                              << ", diff=" << diff << "\n";
                }
                if (diff > max_diff) max_diff = diff;
            }
        }
        std::cout << "Max matrix difference: " << max_diff << "\n";
    }

    // Free device memory
    if (mode != MODE_CPU_ONLY) {
        cudaFree(d_A);
        cudaFree(d_B);
    }
    
    return 0;
}