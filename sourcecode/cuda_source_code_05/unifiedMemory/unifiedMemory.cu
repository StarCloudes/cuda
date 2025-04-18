//=============================================================================================
// Name				: unifiedMemory.cu
// Author	  		: Jose Refojo
// Version	 		:	12-04-2018
// Creation date	:	12-04-2018
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use cuda's Unified Memory to add two vectors
//=============================================================================================

#include "stdio.h"

#define N 18

// This is another possibility, which is even easier to write and use
__device__ __managed__  float  d_unified[N];

__global__ void add_arrays_unified( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		out[idx]=in1[idx]+in2[idx];
	}
}

int main() {
	// pointers to unified memory
	float *a_unified, *b_unified, *c_unified;
	int i;

	// Allocate arrays a_unified, b_unified and c_unified on device
	cudaMallocManaged(&a_unified, sizeof(float)*N);
	cudaMallocManaged(&b_unified, sizeof(float)*N);
	cudaMallocManaged(&c_unified, sizeof(float)*N);

	// Initialize arrays a and b
	for (i=0; i<N; i++) {
		a_unified[i]= (float) i;
		b_unified[i]= (float) 2*i;
	}

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	add_arrays_unified<<<dimGrid,dimBlock>>>(a_unified, b_unified, c_unified, N);
	add_arrays_unified<<<dimGrid,dimBlock>>>(a_unified, c_unified, d_unified, N);

	// Try to comment this out and see if still works, I dare you!
	// Another alternative, to simplify the code even further, is to set the global variable: CUDA_LAUNCH_BLOCKING=1
	// So: export "CUDA_LAUNCH_BLOCKING=1"
	cudaDeviceSynchronize();

	// Print results
	printf("addVectorsfloat will generate two vectors, move them to the global memory, and add them together in the GPU\n");
	for (i=0; i<N; i++) {
		printf(" a[%2d](%10f) + b[%2d](%10f) = c_unified[%2d](%10f) d_unified[%2d](%10f)\n",i,a_unified[i],i,b_unified[i],i,c_unified[i],i,d_unified[i]);
	}

	// Free the memory
	cudaFree(a_unified); cudaFree(b_unified);cudaFree(c_unified);
}
