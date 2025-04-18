//=============================================================================================
// Name        		: cudaClock.cu
// Author      		: Jose Refojo
// Version     		:	24-03-14
// Creation date	:	24-03-14
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will show how to use the clock and clock64 cuda function calls
//=============================================================================================

#include "stdio.h"

#define BLOCK_SIZE 8

__device__ float globalMemoryThread[BLOCK_SIZE];

__global__ void cudaClock(	float *measureLocalSingle,double *measureLocalDouble,
				float *measureSharedSingle,double *measureSharedDouble,
				float *measureGlobalSingle,double *measureGlobalDouble,
			 	int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		float localMemoryThread;
		__shared__ float sharedMemoryThread[BLOCK_SIZE];

		localMemoryThread = threadIdx.x;
		sharedMemoryThread[threadIdx.x] = threadIdx.x;
		globalMemoryThread[threadIdx.x] = threadIdx.x;
		
		// Measure load-compute-store in the local memory
		clock_t initialTimeLocalSingle=clock();
		localMemoryThread++;
		measureLocalSingle[idx]=(float)(clock()-initialTimeLocalSingle);
		long long int initialTimeLocalDouble = clock64();
		localMemoryThread++;
		measureLocalDouble[idx]=(double)(clock64()-initialTimeLocalDouble);

		// Measure load-compute-store in the shared memory
		clock_t initialTimeSharedSingle=clock();
		sharedMemoryThread[threadIdx.x]++;
		measureSharedSingle[idx]=(float)(clock()-initialTimeSharedSingle);
		long long int initialTimeSharedDouble = clock64();
		sharedMemoryThread[threadIdx.x]++;
		measureSharedDouble[idx]=(double)(clock64()-initialTimeSharedDouble);

		// Measure load-compute-store in the global memory
		clock_t initialTimeGlobalSingle=clock();
		globalMemoryThread[threadIdx.x]++;
		measureGlobalSingle[idx]=(float)(clock()-initialTimeGlobalSingle);
		long long int initialTimeGlobalDouble = clock64();
		globalMemoryThread[threadIdx.x]++;
		measureGlobalDouble[idx]=(double)(clock64()-initialTimeGlobalDouble);
	}
}

int main() {
	// pointers to host memory
	float *clockLocalSingle,*clockSharedSingle,*clockGlobalSingle;
	double *clockLocalDouble,*clockSharedDouble,*clockGlobalDouble;
	// pointers to device memory
	float *clockLocalSingle_d,*clockSharedSingle_d,*clockGlobalSingle_d;
	double *clockLocalDouble_d,*clockSharedDouble_d,*clockGlobalDouble_d;
	int N=18;
	int i;

	// Allocate arrays on host
	clockLocalSingle  = (float*) malloc(N*sizeof(float));
	clockSharedSingle = (float*) malloc(N*sizeof(float));
	clockGlobalSingle = (float*) malloc(N*sizeof(float));
	clockLocalDouble  = (double*) malloc(N*sizeof(double));
	clockSharedDouble = (double*) malloc(N*sizeof(double));
	clockGlobalDouble = (double*) malloc(N*sizeof(double));

	// Allocate arrays on device
	cudaMalloc ((void **) &clockLocalSingle_d , sizeof(float)*N);
	cudaMalloc ((void **) &clockSharedSingle_d, sizeof(float)*N);
	cudaMalloc ((void **) &clockGlobalSingle_d, sizeof(float)*N);
	cudaMalloc ((void **) &clockLocalDouble_d , sizeof(double)*N);
	cudaMalloc ((void **) &clockSharedDouble_d, sizeof(double)*N);
	cudaMalloc ((void **) &clockGlobalDouble_d, sizeof(double)*N);

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	cudaClock<<<dimGrid,dimBlock>>>(	clockLocalSingle_d ,clockLocalDouble_d,
						clockSharedSingle_d,clockSharedDouble_d,
						clockGlobalSingle_d,clockGlobalDouble_d,
						N);

	// Copy data from device memory to host memory
	cudaMemcpy(clockLocalSingle , clockLocalSingle_d , sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(clockSharedSingle, clockSharedSingle_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(clockGlobalSingle, clockGlobalSingle_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(clockLocalDouble , clockLocalDouble_d , sizeof(double)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(clockSharedDouble, clockSharedDouble_d, sizeof(double)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(clockGlobalDouble, clockGlobalDouble_d, sizeof(double)*N, cudaMemcpyDeviceToHost);

	// Print clockLocalSingle
	printf("cudaClockfloat will get some time measures in the GPU\n");
	for (i=0; i<N; i++) {
		printf("LocalSingle[%2d](%10f)  ",i,clockLocalSingle[i]);
		printf("LocalDouble[%2d](%10f)  ",i,clockLocalDouble[i]);
		printf("SharedSingle[%2d](%10f)  ",i,clockSharedSingle[i]);
		printf("SharedDouble[%2d](%10f)  ",i,clockSharedDouble[i]);
		printf("GlobalSingle[%2d](%10f)  ",i,clockGlobalSingle[i]);
		printf("GlobalDouble[%2d](%10f)\n",i,clockGlobalDouble[i]);
	}

	// Free the memory
	free(clockLocalSingle);	free(clockSharedSingle); free(clockGlobalSingle);
	free(clockLocalDouble); free(clockSharedDouble); free(clockGlobalDouble);
	cudaFree(clockLocalSingle_d); cudaFree(clockSharedSingle_d); cudaFree(clockGlobalSingle_d);
	cudaFree(clockLocalDouble_d); cudaFree(clockSharedDouble_d); cudaFree(clockGlobalDouble_d);
}
