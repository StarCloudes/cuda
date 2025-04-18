#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <cuda_runtime.h>

//==================================================
// Utility functions
//==================================================

// Get time in microseconds for CPU timing
static inline double get_time_in_microseconds()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// Check for CUDA errors
static inline void checkCudaError(const char* message)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

//==================================================
// CPU Reference Implementations
//==================================================

// rowSumAbs: sum of absolute values per row -> rowSums[n]
void rowSumAbsCPU(const float* matrix, float* rowSums, int n, int m)
{
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        rowSums[i] = sum;
    }
}

// colSumAbs: sum of absolute values per column -> colSums[m]
void colSumAbsCPU(const float* matrix, float* colSums, int n, int m)
{
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        colSums[j] = sum;
    }
}

// reduceCPU: naive sum of vector
float reduceCPU(const float* vec, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += vec[i];
    }
    return sum;
}

//==================================================
// GPU Kernels
//==================================================

// Kernel for rowSumAbs: each row handled by one thread (for simplicity)
__global__ void kernel_rowSumAbs(const float* d_matrix, float* d_rowSums, int n, int m)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            sum += fabsf(d_matrix[row*m + j]);
        }
        d_rowSums[row] = sum;
    }
}

// Kernel for colSumAbs: each column handled by one thread (for simplicity)
__global__ void kernel_colSumAbs(const float* d_matrix, float* d_colSums, int n, int m)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < m) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += fabsf(d_matrix[i*m + col]);
        }
        d_colSums[col] = sum;
    }
}

// Naive reduce kernel: single-thread approach for simplicity (no shared mem / no atomic)
__global__ void kernel_reduce(const float* d_vec, float* d_result, int len)
{
    // Single-thread reduction
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            sum += d_vec[i];
        }
        d_result[0] = sum;
    }
}

//==================================================
// Main
//==================================================
int main(int argc, char* argv[])
{
    // Defaults
    int n = 10;  // rows
    int m = 10;  // columns
    bool randomSeed = false;  // -r
    bool printCpuTiming = false;  // -t
    bool printGpuTiming = false;  // -gt

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            m = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r") == 0) {
            randomSeed = true;
        }
        else if (strcmp(argv[i], "-t") == 0) {
            printCpuTiming = true;
        }
        else if (strcmp(argv[i], "-gt") == 0) {
            printGpuTiming = true;
        }
    }

    // Seed drand48
    if (randomSeed) {
        struct timeval myRandom;
        gettimeofday(&myRandom, NULL);
        srand48((long)myRandom.tv_usec);
    } else {
        srand48(1234567);
    }

    // Allocate host memory for matrix
    float* h_matrix = new float[n*m];

    // Initialize matrix with random values in [-20, 20]
    for (int i = 0; i < n*m; i++) {
        h_matrix[i] = static_cast<float>(drand48() * 40.0 - 20.0);
    }

    // Allocate host memory for CPU results
    float* h_rowSums_cpu = new float[n];
    float* h_colSums_cpu = new float[m];
    float rowReduced_cpu, colReduced_cpu;

    // CPU Timings
    double start_cpu, end_cpu;
    double rowSumTime_cpu = 0.0, colSumTime_cpu = 0.0;
    double rowReduceTime_cpu = 0.0, colReduceTime_cpu = 0.0;

    // 1. Row sums (CPU)
    start_cpu = get_time_in_microseconds();
    rowSumAbsCPU(h_matrix, h_rowSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    rowSumTime_cpu = end_cpu - start_cpu;

    // 2. Column sums (CPU)
    start_cpu = get_time_in_microseconds();
    colSumAbsCPU(h_matrix, h_colSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    colSumTime_cpu = end_cpu - start_cpu;

    // 3. Reduce row sums (CPU)
    start_cpu = get_time_in_microseconds();
    rowReduced_cpu = reduceCPU(h_rowSums_cpu, n);
    end_cpu = get_time_in_microseconds();
    rowReduceTime_cpu = end_cpu - start_cpu;

    // 4. Reduce column sums (CPU)
    start_cpu = get_time_in_microseconds();
    colReduced_cpu = reduceCPU(h_colSums_cpu, m);
    end_cpu = get_time_in_microseconds();
    colReduceTime_cpu = end_cpu - start_cpu;

    //--------------------------------------------------
    // GPU Implementation
    //--------------------------------------------------

    // Allocate device memory
    float *d_matrix, *d_rowSums, *d_colSums, *d_tempResult;
    cudaMalloc((void**)&d_matrix,   n*m*sizeof(float));
    cudaMalloc((void**)&d_rowSums,  n*sizeof(float));
    cudaMalloc((void**)&d_colSums,  m*sizeof(float));
    cudaMalloc((void**)&d_tempResult, sizeof(float)); // for reduce results
    checkCudaError("malloc");

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, n*m*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("memcpy to device");

    // Set up GPU timers
    cudaEvent_t startEvent, endEvent;
    float rowSumTime_gpu  = 0.0f;
    float colSumTime_gpu  = 0.0f;
    float rowReduceTime_gpu  = 0.0f;
    float colReduceTime_gpu  = 0.0f;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    // We use a simple 1D grid: for row/col summation kernels.
    int blockSizeRows = 256;  
    int gridSizeRows  = (n + blockSizeRows - 1) / blockSizeRows;

    int blockSizeCols = 256;
    int gridSizeCols  = (m + blockSizeCols - 1) / blockSizeCols;

    //--------------------------------------------------
    // 1. GPU Row Summation
    //--------------------------------------------------
    cudaEventRecord(startEvent);
    kernel_rowSumAbs<<<gridSizeRows, blockSizeRows>>>(d_matrix, d_rowSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_rowSumAbs");
    cudaEventElapsedTime(&rowSumTime_gpu, startEvent, endEvent);

    //--------------------------------------------------
    // 2. GPU Column Summation
    //--------------------------------------------------
    cudaEventRecord(startEvent);
    kernel_colSumAbs<<<gridSizeCols, blockSizeCols>>>(d_matrix, d_colSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_colSumAbs");
    cudaEventElapsedTime(&colSumTime_gpu, startEvent, endEvent);

    //--------------------------------------------------
    // 3. GPU Reduce for row sums
    //--------------------------------------------------
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_rowSums, d_tempResult, n);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_reduce(row)");
    float rowReduced_gpu = 0.0f;
    cudaMemcpy(&rowReduced_gpu, d_tempResult, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&rowReduceTime_gpu, startEvent, endEvent);

    //--------------------------------------------------
    // 4. GPU Reduce for column sums
    //--------------------------------------------------
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_colSums, d_tempResult, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_reduce(col)");
    float colReduced_gpu = 0.0f;
    cudaMemcpy(&colReduced_gpu, d_tempResult, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&colReduceTime_gpu, startEvent, endEvent);

    //--------------------------------------------------
    // Copy rowSums and colSums back to host for validation (optional)
    //--------------------------------------------------
    float* h_rowSums_gpu = new float[n];
    float* h_colSums_gpu = new float[m];
    cudaMemcpy(h_rowSums_gpu, d_rowSums, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colSums_gpu, d_colSums, m*sizeof(float), cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // Print results
    //--------------------------------------------------
    std::cout << "Matrix size: " << n << " x " << m << std::endl;
    std::cout << "CPU row-reduced = " << rowReduced_cpu << ", GPU row-reduced = " << rowReduced_gpu << std::endl;
    std::cout << "CPU col-reduced = " << colReduced_cpu << ", GPU col-reduced = " << colReduced_gpu << std::endl;

    // Timings
    if (printCpuTiming) {
        std::cout << "\n[CPU Timings in microseconds]\n";
        std::cout << "  Row Summation : " << rowSumTime_cpu << std::endl;
        std::cout << "  Col Summation : " << colSumTime_cpu << std::endl;
        std::cout << "  Row Reduce    : " << rowReduceTime_cpu << std::endl;
        std::cout << "  Col Reduce    : " << colReduceTime_cpu << std::endl;
    }

    if (printGpuTiming) {
        std::cout << "\n[GPU Timings in milliseconds]\n";
        std::cout << "  Row Summation : " << rowSumTime_gpu << std::endl;
        std::cout << "  Col Summation : " << colSumTime_gpu << std::endl;
        std::cout << "  Row Reduce    : " << rowReduceTime_gpu << std::endl;
        std::cout << "  Col Reduce    : " << colReduceTime_gpu << std::endl;
    }

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------
    delete[] h_matrix;
    delete[] h_rowSums_cpu;
    delete[] h_colSums_cpu;
    delete[] h_rowSums_gpu;
    delete[] h_colSums_gpu;

    cudaFree(d_matrix);
    cudaFree(d_rowSums);
    cudaFree(d_colSums);
    cudaFree(d_tempResult);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    return 0;
}