//=============================================================================================
// Name        		: dynamicParalellism.cu
// Author      		: Jose Refojo
// Version     		:	03-03-14
// Creation date	:	19-03-14
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will show how to spawn a child grid from a parent grid using dynamic paralellism
//=============================================================================================

#include "stdio.h"


__global__ void child_kernel ( float *in1, float *in2, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<Ntot/2)
		out[idx]+=10;
}
__global__ void child_kernel2 ( float *in1, float *in2, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
	out[idx]+=20;
}


__device__ void child_kernel_device ( float *in1, float *in2, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;

	if (idx==0) {
		// We can't call child kernels from __device__ functions - and yet the compiler does not complaint about this:
		child_kernel<<<16, 1>>>(in1,in2,out,Ntot);
	}
} 

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		out[idx]=in1[idx]+in2[idx];
		//out[idx]=10;
		if (idx==0) {
			child_kernel<<<16, 1>>>(in1,in2,out,Ntot);
			// The compiler does not complaint about this, but cuda-memcheck does!
			//child_kernel_device(in1,in2,out,Ntot);
		} else if (idx==1) {
			child_kernel2<<<16, 1>>>(in1,in2,out,Ntot);
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
	add_arrays_gpu<<<dimGrid,dimBlock>>>(a_d, b_d, c_d, N);

	// Copy data from deveice memory to host memory
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
