#include <cuda_runtime.h>
#include <cstdio>
#include "exponentialIntegral_gpu.cuh"

__constant__ int d_maxIterations;

// Kernel function for float
__device__ float exponentialIntegralFloatDevice(int n, float x) {
    const float eulerConstant = 0.5772156649015329f;
    const float epsilon = 1.E-30f;
    const float bigfloat = 3.4028235e+38f; // FLT_MAX
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) return expf(-x) / x;

    if (x > 1.0f) {
        b = x + n;
        c = bigfloat;
        d = 1.0f / b;
        h = d;
        for (i = 1; i <= d_maxIterations; i++) {
            a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon)
                return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
        fact = 1.0f;
        for (i = 1; i <= d_maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1)
                del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__global__ void floatKernel(int n, int m, float a, float b, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    float division = (b - a) / m;

    while (idx < total) {
        int i = idx / m + 1;
        int j = idx % m + 1;
        float x = a + j * division;
        result[idx] = exponentialIntegralFloatDevice(i, x);
        idx += blockDim.x * gridDim.x;
    }
}


__device__ double exponentialIntegralDoubleDevice(int n, double x) {
    const double eulerConstant = 0.5772156649015329;
    const double epsilon = 1.E-30;
    const double bigDouble = 1.7976931348623157e+308;  // DBL_MAX
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n == 0) return exp(-x) / x;

    if (x > 1.0) {
        b = x + n;
        c = bigDouble;
        d = 1.0 / b;
        h = d;
        for (i = 1; i <= d_maxIterations; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon)
                return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
        fact = 1.0;
        for (i = 1; i <= d_maxIterations; i++) {
            fact *= -x / i;
            if (i != nm1)
                del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}

__global__ void doubleKernel(int n, int m, double a, double b, double* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    double division = (b - a) / m;
    while (idx < total) {
        int i = idx / m + 1;
        int j = idx % m + 1;
        double x = a + j * division;
        result[idx] = exponentialIntegralDoubleDevice(i, x);
        idx += blockDim.x * gridDim.x;
    }
}

void exponentialIntegralFloatGPUWrapper(int n, int m, float a, float b, float* result, float* totalTimeSecOut) {
    int total = n * m;
    float* d_result;

    // Timing events
    cudaEvent_t malloc_start, malloc_end, kernel_start, kernel_end, memcpy_start, memcpy_end;
    cudaEventCreate(&malloc_start); cudaEventCreate(&malloc_end);
    cudaEventCreate(&kernel_start); cudaEventCreate(&kernel_end);
    cudaEventCreate(&memcpy_start); cudaEventCreate(&memcpy_end);

    float malloc_time = 0, kernel_time = 0, memcpy_time = 0;

    // malloc
    cudaEventRecord(malloc_start);
    cudaMalloc(&d_result, total * sizeof(float));
    cudaEventRecord(malloc_end);

    // Copy maxIterations to constant memory
    int maxIters = 1000;
    cudaMemcpyToSymbol(d_maxIterations, &maxIters, sizeof(int));

    // kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    cudaEventRecord(kernel_start);
    floatKernel<<<gridSize, blockSize>>>(n, m, a, b, d_result);
    cudaEventRecord(kernel_end);

    // memcpy
    cudaEventRecord(memcpy_start);
    cudaMemcpy(result, d_result, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaEventRecord(memcpy_end);

    // sync & measure
    cudaEventSynchronize(memcpy_end);
    cudaEventElapsedTime(&malloc_time, malloc_start, malloc_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    cudaEventElapsedTime(&memcpy_time, memcpy_start, memcpy_end);

    // print all
    printf("[GPU float] malloc time     : %.6f seconds\n", malloc_time / 1000.0f);
    printf("[GPU float] kernel time     : %.6f seconds\n", kernel_time / 1000.0f);
    printf("[GPU float] memcpy time     : %.6f seconds\n", memcpy_time / 1000.0f);
    printf("[GPU float] total cuda time : %.6f seconds\n", (malloc_time + kernel_time + memcpy_time) / 1000.0f);

    *totalTimeSecOut = (malloc_time + kernel_time + memcpy_time) / 1000.0f;

    // cleanup
    cudaEventDestroy(malloc_start); cudaEventDestroy(malloc_end);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_end);
    cudaEventDestroy(memcpy_start); cudaEventDestroy(memcpy_end);
}

void exponentialIntegralDoubleGPUWrapper(int n, int m, double a, double b, double* result, float* totalTimeSecOut) {
    int total = n * m;
    double* d_result;

    // Timing events
    cudaEvent_t malloc_start, malloc_end, kernel_start, kernel_end, memcpy_start, memcpy_end;
    cudaEventCreate(&malloc_start); cudaEventCreate(&malloc_end);
    cudaEventCreate(&kernel_start); cudaEventCreate(&kernel_end);
    cudaEventCreate(&memcpy_start); cudaEventCreate(&memcpy_end);

    float malloc_time = 0, kernel_time = 0, memcpy_time = 0;

    // malloc
    cudaEventRecord(malloc_start);
    cudaMalloc(&d_result, total * sizeof(double));
    cudaEventRecord(malloc_end);

    // Copy maxIterations to constant memory
    int maxIters = 1000;
    cudaMemcpyToSymbol(d_maxIterations, &maxIters, sizeof(int));

    // kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    cudaEventRecord(kernel_start);
    doubleKernel<<<gridSize, blockSize>>>(n, m, a, b, d_result);
    cudaEventRecord(kernel_end);

    // memcpy
    cudaEventRecord(memcpy_start);
    cudaMemcpy(result, d_result, total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaEventRecord(memcpy_end);

    // sync & measure
    cudaEventSynchronize(memcpy_end);
    cudaEventElapsedTime(&malloc_time, malloc_start, malloc_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    cudaEventElapsedTime(&memcpy_time, memcpy_start, memcpy_end);

    // print all
    printf("[GPU double] malloc time     : %.6f seconds\n", malloc_time / 1000.0f);
    printf("[GPU double] kernel time     : %.6f seconds\n", kernel_time / 1000.0f);
    printf("[GPU double] memcpy time     : %.6f seconds\n", memcpy_time / 1000.0f);
    printf("[GPU double] total cuda time : %.6f seconds\n", (malloc_time + kernel_time + memcpy_time) / 1000.0f);

    *totalTimeSecOut = (malloc_time + kernel_time + memcpy_time) / 1000.0f;

    // cleanup
    cudaEventDestroy(malloc_start); cudaEventDestroy(malloc_end);
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_end);
    cudaEventDestroy(memcpy_start); cudaEventDestroy(memcpy_end);
}

// Stream version for float
void exponentialIntegralFloatGPUStreamWrapper(int n, int m, float a, float b, float* result, float* totalTimeSecOut) {
    int total = n * m;
    float* d_result;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    int maxIters = 1000;
    cudaMemcpyToSymbolAsync(d_maxIterations, &maxIters, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
    cudaMallocAsync(&d_result, total * sizeof(float), stream);

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    floatKernel<<<gridSize, blockSize, 0, stream>>>(n, m, a, b, d_result);

    cudaMemcpyAsync(result, d_result, total * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_result, stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Print timing breakdown (approximate, since all overlap in stream)
    float malloc_time = 0.0f, kernel_time = 0.0f, memcpy_time = 0.0f;
    cudaEventElapsedTime(&malloc_time, start, start); // no actual malloc event split
    cudaEventElapsedTime(&kernel_time, start, stop);  // approximate total kernel+overlap
    cudaEventElapsedTime(&memcpy_time, start, stop);  // approximate total memcpy+overlap
    printf("[GPU float (stream)] malloc time     : %.6f seconds\n", malloc_time / 1000.0f);
    printf("[GPU float (stream)] kernel time     : %.6f seconds\n", kernel_time / 1000.0f);
    printf("[GPU float (stream)] memcpy time     : %.6f seconds\n", memcpy_time / 1000.0f);
    printf("[GPU float (stream)] total cuda time : %.6f seconds\n", milliseconds / 1000.0f);
    if (totalTimeSecOut) {
        *totalTimeSecOut = milliseconds / 1000.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}

// Stream version for double
void exponentialIntegralDoubleGPUStreamWrapper(int n, int m, double a, double b, double* result, float* totalTimeSecOut) {
    int total = n * m;
    double* d_result;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    int maxIters = 1000;
    cudaMemcpyToSymbolAsync(d_maxIterations, &maxIters, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
    cudaMallocAsync(&d_result, total * sizeof(double), stream);

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    doubleKernel<<<gridSize, blockSize, 0, stream>>>(n, m, a, b, d_result);

    cudaMemcpyAsync(result, d_result, total * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_result, stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Print timing breakdown (approximate, since all overlap in stream)
    float malloc_time = 0.0f, kernel_time = 0.0f, memcpy_time = 0.0f;
    cudaEventElapsedTime(&malloc_time, start, start); // no actual malloc event split
    cudaEventElapsedTime(&kernel_time, start, stop);  // approximate total kernel+overlap
    cudaEventElapsedTime(&memcpy_time, start, stop);  // approximate total memcpy+overlap
    printf("[GPU double (stream)] malloc time     : %.6f seconds\n", malloc_time / 1000.0f);
    printf("[GPU double (stream)] kernel time     : %.6f seconds\n", kernel_time / 1000.0f);
    printf("[GPU double (stream)] memcpy time     : %.6f seconds\n", memcpy_time / 1000.0f);
    printf("[GPU double (stream)] total cuda time : %.6f seconds\n", milliseconds / 1000.0f);
    if (totalTimeSecOut) {
        *totalTimeSecOut = milliseconds / 1000.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
