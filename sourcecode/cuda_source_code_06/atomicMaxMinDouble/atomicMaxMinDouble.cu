//=============================================================================================
// Name        		: atomicMaxMindouble.cu
// Author      		: Jose Refojo
// Version     		:	05-03-2013
// Creation date	:	20-02-2019
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize an array of doubles and then apply atomic
//				min and max operations on it
//=============================================================================================

#include "stdio.h"

#include <iostream>

using namespace std;
using std::cout;
using std::endl;

bool checkCudaErrors(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n",msg,cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
		return (false);
	}   
	return (true);
}

__device__ double atomicAddDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__device__ double atomicMaxDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(max(val,__longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__device__ double atomicMinDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(min(val,__longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void apply_atomic_operation( double *input, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		atomicAddDouble (&(input[idx]),1.5f);
		//atomicMaxDouble (&(input[idx]),5.5f);
		//atomicMinDouble (&(input[idx]),5.5f);
	}
}

int main() {
	int N=10;
	int i;

	// pointers to host memory
	double *input_host, *output_host;
	// pointers to device memory
	double *input_device;

	// Allocate arrays host
	input_host = (double*) malloc(N*sizeof(double));
	output_host = (double*) malloc(N*sizeof(double));

	// Allocate arrays on device
	cudaMalloc ((void **) &input_device, sizeof(double)*N);

	// Initialize arrays
	for (i=0; i<N; i++) {
		input_host[i]= (double) (i);
		output_host[i]= 0;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(input_device, input_host, sizeof(double)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	apply_atomic_operation<<<dimGrid,dimBlock>>>(input_device, N);

	// Copy data from device memory to host memory
	cudaMemcpy(output_host, input_device, sizeof(double)*N, cudaMemcpyDeviceToHost);

	// Print output_host
	for (i=0; i<N; i++) {
		printf("input_host[%d]=%f +1.5 should be = output_host[%d]=%f\n",i,input_host[i],i,output_host[i]);
	}

	// Free the memory
	free(input_host);
	free(output_host);
	cudaFree(input_device);

}
