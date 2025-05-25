///// CUDA implementation header for exponential integral calculation
///// Created for MAP55616-03 assignment
//------------------------------------------------------------------------------
// File : exponential_integral_cuda.h
//------------------------------------------------------------------------------

#ifndef EXPONENTIAL_INTEGRAL_CUDA_H
#define EXPONENTIAL_INTEGRAL_CUDA_H

#include <vector>

// Main CUDA computation function
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
);

// Advanced version supporting shared memory, streams, and texture memory
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
);

#endif // EXPONENTIAL_INTEGRAL_CUDA_H