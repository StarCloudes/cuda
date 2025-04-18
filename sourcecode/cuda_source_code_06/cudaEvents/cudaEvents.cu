//=============================================================================================
// Name        		: cudaEvents.cu
// Author      		: Jose Refojo
// Version     		:	22-02-2013
// Creation date	:	22-02-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use cudaEvents
//=============================================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
 #include <sys/time.h>

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		for (int i=0;i<100000;i++) {
			if (idx%2) {
			       out[idx]=0.12345*in1[idx]+123.456*in2[idx]+123.987*threadIdx.x;
			} else {
			       out[idx]=-0.12345*in1[idx]-123.456*in2[idx]-123.987*threadIdx.x;
			}
		}
	}
}

int main() {
	// pointers to host memory
	float *a, *b, *c;
	// pointers to device memory
	float *a_d, *b_d, *c_d;
	int N=1000000;
	int numberOfIterations=5000;
	int numberOfTests=100;
	int i,j;

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

	// Compute the execution configuration
	int block_size=1;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Initialise all the time measuring
	cudaEvent_t start, finish;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	struct timeval timeValKernelStart, timeValKernelEnd;

	clock_t kernelStart,kernelEnd;

	long int problemSize;
	for (i=0;i<numberOfTests;i++) {
		// Start measuring
		cudaEventRecord(start, 0); // We use 0 here because it is the "default" stream
		kernelStart = clock();
		gettimeofday(&timeValKernelStart, NULL);

		// Copy data from host memory to device memory
		cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

		// Call the kernel
		for (j=1;j<numberOfIterations*i;j++)
			add_arrays_gpu<<<dimGrid,dimBlock,0,0>>>(a_d, b_d, c_d, N);

		// Copy data from deveice memory to host memory
		cudaMemcpy(c, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

		// End measuring
		cudaEventRecord(finish, 0);
		kernelEnd=clock();
		gettimeofday(&timeValKernelEnd, NULL);

		// Calculate and output times
		cudaEventSynchronize(start);  // This is optional, we shouldn't need it
		cudaEventSynchronize(finish); // This isn't - we need to wait for the event to finish
		cudaEventElapsedTime(&elapsedTime, start, finish);
		float kernelTotal=(float)(kernelEnd-kernelStart)/(float)(CLOCKS_PER_SEC);
		float timeValTotal = (float)((timeValKernelEnd.tv_sec * 1000000 + timeValKernelEnd.tv_usec) - (timeValKernelStart.tv_sec * 1000000 + timeValKernelStart.tv_usec));

		problemSize = (long int)(numberOfIterations)*(long int)(i);
		printf("Amount of time taken by the kernel when running problem size %ld: (event)= %f (clock)=%f (timeval)=%f\n",problemSize,elapsedTime,kernelTotal,timeValTotal);
	}

	// Print c
//	for (i=0; i<5; i++) {
//		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
//	}

	// Free the memory
	free(a); free(b); free(c);
	cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);
}
