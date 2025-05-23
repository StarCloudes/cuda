#include <cuda_runtime.h>
#include <cstdio>
#include "exponentialIntegral_gpu.cuh"

// Kernel function for float
__device__ float exponentialIntegralFloatDevice(int n, float x, int maxIterations) {
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
        for (i = 1; i <= maxIterations; i++) {
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
        for (i = 1; i <= maxIterations; i++) {
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

__global__ void floatKernel(int n, int m, float a, float b, float* result, int maxIterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    float division = (b - a) / m;

    while (idx < total) {
        int i = idx / m + 1;
        int j = idx % m + 1;
        float x = a + j * division;
        result[idx] = exponentialIntegralFloatDevice(i, x, maxIterations);
        idx += blockDim.x * gridDim.x;
    }
}


__device__ double exponentialIntegralDoubleDevice(int n, double x, int maxIterations) {
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
        for (i = 1; i <= maxIterations; i++) {
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
        for (i = 1; i <= maxIterations; i++) {
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

__global__ void doubleKernel(int n, int m, double a, double b, double* result, int maxIterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    double division = (b - a) / m;
    while (idx < total) {
        int i = idx / m + 1;
        int j = idx % m + 1;
        double x = a + j * division;
        result[idx] = exponentialIntegralDoubleDevice(i, x, maxIterations);
        idx += blockDim.x * gridDim.x;
    }
}

void exponentialIntegralFloatGPUWrapper(int n, int m, float a, float b, float* result) {
    int total = n * m;
    float* d_result;
    cudaMalloc(&d_result, total * sizeof(float));

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    floatKernel<<<gridSize, blockSize>>>(n, m, a, b, d_result, 1000); // maxIter

    cudaMemcpy(result, d_result, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

void exponentialIntegralDoubleGPUWrapper(int n, int m, double a, double b, double* result) {
    double* d_result;
    cudaMalloc(&d_result, n * m * sizeof(double));
    doubleKernel<<<(n * m + 255)/256, 256>>>(n, m, a, b, d_result, 1000);
    cudaMemcpy(result, d_result, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}