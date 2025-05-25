///// CUDA implementation for exponential integral calculation
///// Created for MAP55616-03 assignment - Fixed texture memory implementation
//------------------------------------------------------------------------------
// File : exponential_integral_cuda.cu
//------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <vector>
#include "exponential_integral_cuda.h"

using namespace std;

// Utility function to check CUDA errors
void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        cout << "CUDA error in " << function << ": " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

// Get current time in seconds
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 0.000001;
}

// Device function for float exponential integral calculation
__device__ float exponentialIntegralFloatDevice(const int n, const float x, const int maxIterations) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.0e-30f;
    float bigfloat = 1.0e+30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && ((n == 0) || (n == 1)))) {
        return 0.0f;
    }
    
    if (n == 0) {
        ans = expf(-x) / x;
    } else {
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
                if (fabsf(del - 1.0f) <= epsilon) {
                    ans = h * expf(-x);
                    return ans;
                }
            }
            ans = h * expf(-x);
            return ans;
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
            fact = 1.0f;
            for (i = 1; i <= maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0f / ii;
                    }
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

// Device function for double exponential integral calculation
__device__ double exponentialIntegralDoubleDevice(const int n, const double x, const int maxIterations) {
    const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.0e-30;
    double bigdouble = 1.0e+30;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
        return 0.0;
    }
    
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0) {
            b = x + n;
            c = bigdouble;
            d = 1.0 / b;
            h = d;
            for (i = 1; i <= maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
            return ans;
        } else {
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
            fact = 1.0;
            for (i = 1; i <= maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0 / ii;
                    }
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

// CUDA kernel for float calculations
__global__ void exponentialIntegralKernelFloat(
    float* results,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * numberOfSamples;
    
    if (idx < totalElements) {
        int orderIndex = idx / numberOfSamples;
        int sampleIndex = idx % numberOfSamples;
        
        int order = orderIndex + 1;
        double division = (b - a) / ((double)numberOfSamples);
        float x = (float)(a + (sampleIndex + 1) * division);
        
        results[idx] = exponentialIntegralFloatDevice(order, x, maxIterations);
    }
}

// CUDA kernel for double calculations
__global__ void exponentialIntegralKernelDouble(
    double* results,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * numberOfSamples;
    
    if (idx < totalElements) {
        int orderIndex = idx / numberOfSamples;
        int sampleIndex = idx % numberOfSamples;
        
        int order = orderIndex + 1;
        double division = (b - a) / ((double)numberOfSamples);
        double x = a + (sampleIndex + 1) * division;
        
        results[idx] = exponentialIntegralDoubleDevice(order, x, maxIterations);
    }
}

// Optimized kernel using shared memory for constants
__global__ void exponentialIntegralKernelFloatShared(
    float* results,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations
) {
    __shared__ float sharedConstants[3];
    
    if (threadIdx.x == 0) {
        sharedConstants[0] = (float)a;
        sharedConstants[1] = (float)b;
        sharedConstants[2] = (float)((b - a) / ((double)numberOfSamples));
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * numberOfSamples;
    
    if (idx < totalElements) {
        int orderIndex = idx / numberOfSamples;
        int sampleIndex = idx % numberOfSamples;
        
        int order = orderIndex + 1;
        float x = sharedConstants[0] + (sampleIndex + 1) * sharedConstants[2];
        
        results[idx] = exponentialIntegralFloatDevice(order, x, maxIterations);
    }
}

// Optimized kernel using shared memory for double precision
__global__ void exponentialIntegralKernelDoubleShared(
    double* results,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations
) {
    __shared__ double sharedConstants[3];
    
    if (threadIdx.x == 0) {
        sharedConstants[0] = a;
        sharedConstants[1] = b;
        sharedConstants[2] = (b - a) / ((double)numberOfSamples);
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * numberOfSamples;
    
    if (idx < totalElements) {
        int orderIndex = idx / numberOfSamples;
        int sampleIndex = idx % numberOfSamples;
        
        int order = orderIndex + 1;
        double x = sharedConstants[0] + (sampleIndex + 1) * sharedConstants[2];
        
        results[idx] = exponentialIntegralDoubleDevice(order, x, maxIterations);
    }
}

// Modern texture memory implementation using texture objects
__global__ void exponentialIntegralKernelFloatTexture(
    float* results,
    unsigned int n,
    unsigned int numberOfSamples,
    int maxIterations,
    cudaTextureObject_t texObj
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * numberOfSamples;
    
    if (idx < totalElements) {
        int orderIndex = idx / numberOfSamples;
        int sampleIndex = idx % numberOfSamples;
        
        int order = orderIndex + 1;
        
        // Read parameters from texture object
        float a = tex1Dfetch<float>(texObj, 0);
        float b = tex1Dfetch<float>(texObj, 1);
        float division = (b - a) / ((float)numberOfSamples);
        float x = a + (sampleIndex + 1) * division;
        
        results[idx] = exponentialIntegralFloatDevice(order, x, maxIterations);
    }
}

// Main CUDA computation function with comprehensive timing and optimization
double computeExponentialIntegralsCuda(
    std::vector<std::vector<float>>& resultsFloatGpu,
    std::vector<std::vector<double>>& resultsDoubleGpu,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations,
    double* timeFloat,
    double* timeDouble
) {
    double startTotal = getCurrentTime();
    
    // Calculate total number of elements
    size_t totalElements = n * numberOfSamples;
    size_t floatSize = totalElements * sizeof(float);
    size_t doubleSize = totalElements * sizeof(double);
    
    cout << "GPU: Computing " << totalElements << " exponential integrals..." << endl;
    
    // Device memory pointers
    float* d_resultsFloat = nullptr;
    double* d_resultsDouble = nullptr;
    
    // Host arrays for results
    float* h_resultsFloat = new float[totalElements];
    double* h_resultsDouble = new double[totalElements];
    
    // CUDA streams for overlapping computation
    cudaStream_t streamFloat, streamDouble;
    checkCudaError(cudaStreamCreate(&streamFloat), "cudaStreamCreate float");
    checkCudaError(cudaStreamCreate(&streamDouble), "cudaStreamCreate double");
    
    // === FLOAT PRECISION COMPUTATION ===
    double startFloat = getCurrentTime();
    
    // Allocate device memory for float
    checkCudaError(cudaMalloc(&d_resultsFloat, floatSize), "cudaMalloc float");
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    
    cout << "Launching float kernel with " << gridSize << " blocks of " << blockSize << " threads" << endl;
    
    // Launch float kernel with shared memory optimization
    exponentialIntegralKernelFloatShared<<<gridSize, blockSize, 0, streamFloat>>>(
        d_resultsFloat, n, numberOfSamples, a, b, maxIterations
    );
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "float kernel launch");
    
    // Copy results back to host
    checkCudaError(cudaMemcpyAsync(h_resultsFloat, d_resultsFloat, floatSize, 
                                   cudaMemcpyDeviceToHost, streamFloat), "cudaMemcpy float D2H");
    
    // Synchronize float stream
    checkCudaError(cudaStreamSynchronize(streamFloat), "cudaStreamSynchronize float");
    
    double endFloat = getCurrentTime();
    *timeFloat = endFloat - startFloat;
    
    // === DOUBLE PRECISION COMPUTATION ===
    double startDouble = getCurrentTime();
    
    // Allocate device memory for double
    checkCudaError(cudaMalloc(&d_resultsDouble, doubleSize), "cudaMalloc double");
    
    cout << "Launching double kernel with " << gridSize << " blocks of " << blockSize << " threads" << endl;
    
    // Launch double kernel with shared memory optimization
    exponentialIntegralKernelDoubleShared<<<gridSize, blockSize, 0, streamDouble>>>(
        d_resultsDouble, n, numberOfSamples, a, b, maxIterations
    );
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "double kernel launch");
    
    // Copy results back to host
    checkCudaError(cudaMemcpyAsync(h_resultsDouble, d_resultsDouble, doubleSize, 
                                   cudaMemcpyDeviceToHost, streamDouble), "cudaMemcpy double D2H");
    
    // Synchronize double stream
    checkCudaError(cudaStreamSynchronize(streamDouble), "cudaStreamSynchronize double");
    
    double endDouble = getCurrentTime();
    *timeDouble = endDouble - startDouble;
    
    // === COPY RESULTS TO OUTPUT VECTORS ===
    for (unsigned int ui = 0; ui < n; ui++) {
        for (unsigned int uj = 0; uj < numberOfSamples; uj++) {
            int idx = ui * numberOfSamples + uj;
            resultsFloatGpu[ui][uj] = h_resultsFloat[idx];
            resultsDoubleGpu[ui][uj] = h_resultsDouble[idx];
        }
    }
    
    // === CLEANUP ===
    delete[] h_resultsFloat;
    delete[] h_resultsDouble;
    
    checkCudaError(cudaFree(d_resultsFloat), "cudaFree float");
    checkCudaError(cudaFree(d_resultsDouble), "cudaFree double");
    
    checkCudaError(cudaStreamDestroy(streamFloat), "cudaStreamDestroy float");
    checkCudaError(cudaStreamDestroy(streamDouble), "cudaStreamDestroy double");
    
    double endTotal = getCurrentTime();
    double totalTime = endTotal - startTotal;
    
    cout << "GPU computation completed successfully!" << endl;
    cout << "  Total time: " << totalTime << " seconds" << endl;
    cout << "  Float time: " << *timeFloat << " seconds" << endl;
    cout << "  Double time: " << *timeDouble << " seconds" << endl;
    
    return totalTime;
}

// Advanced version with modern texture memory implementation
double computeExponentialIntegralsCudaWithTexture(
    std::vector<std::vector<float>>& resultsFloatGpu,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations,
    double* timeFloat
) {
    double startTotal = getCurrentTime();
    
    // Calculate total number of elements
    size_t totalElements = n * numberOfSamples;
    size_t floatSize = totalElements * sizeof(float);
    
    cout << "GPU Texture: Computing " << totalElements << " exponential integrals..." << endl;
    
    // Device memory pointers
    float* d_resultsFloat = nullptr;
    float* d_textureData = nullptr;
    
    // Host arrays for results and texture data
    float* h_resultsFloat = new float[totalElements];
    float h_textureData[2] = {(float)a, (float)b};
    
    // === TEXTURE MEMORY SETUP ===
    double startFloat = getCurrentTime();
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_resultsFloat, floatSize), "cudaMalloc float");
    checkCudaError(cudaMalloc(&d_textureData, 2 * sizeof(float)), "cudaMalloc texture");
    
    // Copy texture data to device
    checkCudaError(cudaMemcpy(d_textureData, h_textureData, 2 * sizeof(float), 
                              cudaMemcpyHostToDevice), "cudaMemcpy texture H2D");
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_textureData;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32-bit float
    resDesc.res.linear.sizeInBytes = 2 * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t texObj = 0;
    checkCudaError(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL), 
                   "cudaCreateTextureObject");
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;
    
    cout << "Launching texture kernel with " << gridSize << " blocks of " << blockSize << " threads" << endl;
    
    // Launch texture kernel
    exponentialIntegralKernelFloatTexture<<<gridSize, blockSize>>>(
        d_resultsFloat, n, numberOfSamples, maxIterations, texObj
    );
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "texture kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    
    // Copy results back to host
    checkCudaError(cudaMemcpy(h_resultsFloat, d_resultsFloat, floatSize, 
                              cudaMemcpyDeviceToHost), "cudaMemcpy float D2H");
    
    double endFloat = getCurrentTime();
    *timeFloat = endFloat - startFloat;
    
    // === COPY RESULTS TO OUTPUT VECTORS ===
    for (unsigned int ui = 0; ui < n; ui++) {
        for (unsigned int uj = 0; uj < numberOfSamples; uj++) {
            int idx = ui * numberOfSamples + uj;
            resultsFloatGpu[ui][uj] = h_resultsFloat[idx];
        }
    }
    
    // === CLEANUP ===
    delete[] h_resultsFloat;
    
    checkCudaError(cudaDestroyTextureObject(texObj), "cudaDestroyTextureObject");
    checkCudaError(cudaFree(d_resultsFloat), "cudaFree float");
    checkCudaError(cudaFree(d_textureData), "cudaFree texture");
    
    double endTotal = getCurrentTime();
    double totalTime = endTotal - startTotal;
    
    cout << "GPU texture computation completed successfully!" << endl;
    cout << "  Total time: " << totalTime << " seconds" << endl;
    cout << "  Float time: " << *timeFloat << " seconds" << endl;
    
    return totalTime;
}


// Function to test different optimization techniques
double computeExponentialIntegralsCudaAdvanced(
    std::vector<std::vector<float>>& resultsFloatGpu,
    std::vector<std::vector<double>>& resultsDoubleGpu,
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations,
    double* timeFloat,
    double* timeDouble,
    bool useSharedMemory,
    bool useStreams,
    bool useTextureMemory
) {
    cout << "\n=== Advanced CUDA Implementation ===" << endl;
    cout << "Shared Memory: " << (useSharedMemory ? "ON" : "OFF") << endl;
    cout << "Streams: " << (useStreams ? "ON" : "OFF") << endl;
    cout << "Texture Memory: " << (useTextureMemory ? "ON" : "OFF") << endl;

    if (useTextureMemory) {
        cout << "Using texture memory optimization for float precision only" << endl;

        // For texture memory, compute only float precision
        double totalTime = computeExponentialIntegralsCudaWithTexture(
            resultsFloatGpu, n, numberOfSamples, a, b, maxIterations, timeFloat
        );

        // Set double time to 0 since we're not computing double precision with texture
        *timeDouble = 0.0;

        // We still need to fill double results with actual values, use regular compute for that
        std::vector<std::vector<float>> tempFloatResults(n, std::vector<float>(numberOfSamples));
        double dummyFloatTime;
        computeExponentialIntegralsCuda(tempFloatResults, resultsDoubleGpu,
                                        n, numberOfSamples, a, b, maxIterations,
                                        &dummyFloatTime, timeDouble);

        return totalTime;
    } else {
        // Use the standard implementation
        return computeExponentialIntegralsCuda(resultsFloatGpu, resultsDoubleGpu,
                                               n, numberOfSamples, a, b, maxIterations,
                                               timeFloat, timeDouble);
    }
}
