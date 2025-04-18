//=============================================================================================
// Name        		: syncThreadsTest.cu
// Author      		: Jose Refojo
// Version     		:	08-02-2017
// Creation date	:	28-01-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will run a number of block synchronization tests
//=============================================================================================

#include "stdio.h"

__global__ void syncThreadsTest( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		out[idx]=in1[idx]+in2[idx];
		printf ("syncThreadsTest::threadIdx[%d] =%f %d\n",idx,out[idx],idx%2);
			__syncthreads();
		if (idx==0) {
			printf ("\n\n");
		}
		if ((idx%2)==1) {
			__syncthreads();	// This is potentially dangerous!
			printf ("syncThreadsTest::threadIdx[%d] idx%%2=%d\n",idx,idx%2);
		} else {
			printf ("syncThreadsTest::threadIdx[%d] idx%%2=%d\n",idx,idx%2);

		}

		if (idx==0) {
			printf ("\n\n");
		}

		// __syncthreads_count test
		int syncthreads_count = __syncthreads_count(idx%2==0);
		// Another option, that didn't use to work, is to evaluate a variable instead, such as:
		//int even = idx%2;
		//int syncthreads_count = __syncthreads_count(even);
		if (idx==0) {
			printf ("syncThreadsTest::threadIdx[%d] syncthreads_count=%d\n",idx,syncthreads_count);
		}

		// __syncthreads_and test
		int syncthreads_and = __syncthreads_and(idx%2==0);
		if (idx==0) {
			printf ("syncThreadsTest::threadIdx[%d] syncthreads_and=%d\n",idx,syncthreads_and);
		}

		// __syncthreads_or test
		int syncthreads_or = __syncthreads_or(idx%2==0);
		if (idx==0) {
			printf ("syncThreadsTest::threadIdx[%d] syncthreads_or=%d\n",idx,syncthreads_or);
		}
	}
}

int main() {
	// pointers to host memory
	float *a, *b, *c;
	// pointers to device memory
	float *a_d, *b_d, *c_d;
	int N=18;
	int i;

	// Allocate arrays a, b and c on host
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	c = (float*) malloc(N*sizeof(float));

	// Allocate arrays a_d, b_d and c_d on device
	cudaMalloc ((void **) &a_d, sizeof(float)*N);
	cudaMalloc ((void **) &b_d, sizeof(float)*N);
	cudaMalloc ((void **) &c_d, sizeof(float)*N);

	// Initialize arrays a and b
	for (i=0; i<N; i++) {
		a[i]= (float) 2*i;
		b[i]=-(float) i;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	syncThreadsTest<<<dimGrid,dimBlock>>>(a_d, b_d, c_d, N);

	// Copy data from device memory to host memory
	cudaMemcpy(c, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

	// Print c
	printf("addVectorsfloat will generate two vectors, move them to the global memory, and add them together in the GPU\n");
	for (i=0; i<N; i++) {
		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
	}

	// Free the memory
	free(a); free(b); free(c);
	cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);
}
