#include "stdio.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

	if ( idx <Ntot ) {
		// obtain default "current thread block" group
		thread_group my_block = this_thread_block();
		thread_group my_tile = tiled_partition(my_block, 2);

	    // This operation will be performed by only the first 2-thread tile of each block
	    if (my_block.thread_rank() < 2) {
			//out[idx]=100+in1[idx]+in2[idx];
			out[idx]=100*(blockIdx.x+1)+my_block.thread_rank();
    	    my_tile.sync();
    	} else {
			out[idx]=-1.0-(float)(threadIdx.x);
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
		a[i]= (float) i;
		b[i]=+(float) 2*i;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	add_arrays_gpu<<<dimGrid,dimBlock>>>(a_d, b_d, c_d, N);

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
