//=============================================================================================
// Name        		: atomic.cu
// Author      		: Jose Refojo
// Version     		:	28-06-2012
// Creation date	:	27-06-2012
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize an array and then apply an atomic operation on it
//=============================================================================================

#include "stdio.h"

__global__ void apply_atomic_operation( float *input, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		atomicAdd (&(input[idx]),1.5f);
	}
}

int main() {
	// pointers to host memory
	float *input_host, *output_host;
	// pointers to device memory
	float *input_device;
	int N=20;
	int i;

	// Allocate arrays host
	input_host = (float*) malloc(N*sizeof(float));
	output_host = (float*) malloc(N*sizeof(float));

	// Allocate arrays on device
	cudaMalloc ((void **) &input_device, sizeof(float)*N);

	// Initialize arrays
	for (i=0; i<N; i++) {
		input_host[i]= (float) (i);
		output_host[i]= 0;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(input_device, input_host, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	apply_atomic_operation<<<dimGrid,dimBlock>>>(input_device, N);

	// Copy data from device memory to host memory
	cudaMemcpy(output_host, input_device, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print output_host
	for (i=0; i<N; i++) {
		printf("input_host[%d]=%f +1.5 should be = output_host[%d]=%f\n",i,input_host[i],i,output_host[i]);
	}

	// Free the memory
	free(input_host);
	free(output_host);
	cudaFree(input_device);

}
