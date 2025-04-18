//=============================================================================================
// Name        		: globalMemoryDynamic.cu
// Author      		: Jose Refojo
// Version     		:	20-03-2014
// Creation date	:	20-03-2014
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use the dynamically allocated global memory
//=============================================================================================

#define BLOCK_SIZE 8
#include "stdio.h"


__global__ void globalMemoryDynamicPerThread(int *threadIdsGPU,int *threadOtherIdsGPU,int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

	if ( idx <Ntot ) {
		size_t size = Ntot;
		int* array = (int*)malloc(size*sizeof(int));
		memset(array, 0, size*sizeof(int));
		printf("Thread %d got pointer: %p\n", idx, array);

		unsigned int ui;
		for (ui=0;ui<size;ui++) {
			array[ui]=ui+1;
		}

		threadIdsGPU[idx]=0;
		threadOtherIdsGPU[idx]=1;
		blockIdsGPU[idx]=2;

		threadIdsGPU[idx]+=array[0];
		threadOtherIdsGPU[idx]+=array[1];
		blockIdsGPU[idx]+=array[2];

		free(array);
	}
}

__global__ void globalMemoryDynamicPerBlock(int *threadIdsGPU,int *threadOtherIdsGPU,int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

	// array_global is a pointer to dynamically allocated global memory which just happens to sit in the shared memory!
	__shared__ int* array_global;

	if ( idx <Ntot ) {
		size_t size = Ntot;
		if (threadIdx.x==0) {
			int* array = (int*)malloc(size*sizeof(int));
			memset(array, 0, size*sizeof(int));
			printf("Thread %d got pointer: %p\n", idx, array);
			// You could use array_shared directly, but we use array for clarity
			array_global=array;
		}

		__syncthreads();

		printf("Thread %d got pointer: %p\n", idx, array_global);

		array_global[idx]=idx+1;

		threadIdsGPU[idx]=0;
		threadOtherIdsGPU[idx]=1;
		blockIdsGPU[idx]=2;

		threadIdsGPU[idx]+=array_global[0];
		threadOtherIdsGPU[idx]+=array_global[1];
		blockIdsGPU[idx]+=array_global[2];

		__syncthreads();

		if (threadIdx.x==0) {
			// array and array_shared store the same value (the address that we dynamically allocated), so we do not need to keep array
			free(array_global);
		}
	}
}


int main() {
	// pointers to host memory
	int *threadIds,*threadOtherIds, *blockIds;
	// pointers to device memory
	int *threadIdsGPU,*threadOtherIdsGPU, *blockIdsGPU;
	// N is the total size that we want
	int N=18;
	int i;

	// Allocate arrays threadIds and blockIds on host
	threadIds 	= (int*) malloc(N*sizeof(int));
	threadOtherIds 	= (int*) malloc(N*sizeof(int));
	blockIds 	= (int*) malloc(N*sizeof(int));

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIdsGPU, 		sizeof(int)*N);
	cudaMalloc ((void **) &threadOtherIdsGPU,	sizeof(int)*N);
	cudaMalloc ((void **) &blockIdsGPU, 		sizeof(int)*N);

	size_t globalMemoryLimitSize;
	cudaDeviceGetLimit(&globalMemoryLimitSize, cudaLimitMallocHeapSize);
	printf ("The default limit for the malloc heap size is %lu bytes\n",globalMemoryLimitSize);
	// We want more than 8388608 bytes?
	size_t desiredGlobalMemoryLimitSize = 10000000;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, desiredGlobalMemoryLimitSize);

	cudaDeviceGetLimit(&globalMemoryLimitSize, cudaLimitMallocHeapSize);
	printf ("The new limit for the malloc heap size is %lu bytes\n",globalMemoryLimitSize);
	
/*
	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Scan information from the threads
	globalMemoryDynamicPerThread<<<dimGrid,dimBlock>>>(threadIdsGPU,threadOtherIdsGPU, blockIdsGPU, N);
	//globalMemoryDynamicPerBlock<<<dimGrid,dimBlock>>>(threadIdsGPU,threadOtherIdsGPU, blockIdsGPU, N);

	// Copy data from device memory to host memory
	cudaMemcpy(threadIds, threadIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadOtherIds, threadOtherIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockIds, blockIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	if (true) {
		printf(" dimGrid=%d\n",dimGrid.x);
		for (i=0; i<N; i++) {
		       printf(" threadIds[%d]=%d\n",i,threadIds[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" threadOtherIds[%d]=%d\n",i,threadOtherIds[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" blockIds[%d]=%d\n",i,blockIds[i]);
		}
	}

	// Free the memory
	free(threadIds);
	free(threadOtherIds);
	free(blockIds); 

	cudaFree(threadIdsGPU);
	cudaFree(threadOtherIdsGPU);
	cudaFree(blockIdsGPU);
}
