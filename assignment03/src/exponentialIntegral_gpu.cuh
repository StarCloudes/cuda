#ifndef EXPONENTIAL_INTEGRAL_GPU_H
#define EXPONENTIAL_INTEGRAL_GPU_H

void exponentialIntegralFloatGPUWrapper(int n, int m, float a, float b, float* result, float* totalTimeSecOut, int blockSize);
void exponentialIntegralDoubleGPUWrapper(int n, int m, double a, double b, double* result, float* totalTimeSecOut, int blockSize);

void exponentialIntegralFloatGPUStreamWrapper(int n, int m, float a, float b, float* result, float* totalTimeSecOut, int blockSize);
void exponentialIntegralDoubleGPUStreamWrapper(int n, int m, double a, double b, double* result, float* totalTimeSecOut, int blockSize);
#endif