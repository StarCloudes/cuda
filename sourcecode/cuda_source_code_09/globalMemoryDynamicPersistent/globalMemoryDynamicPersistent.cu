//=============================================================================================
// Name        		: globalMemoryDynamicPersistent.cu
// Author      		: Jose Refojo
// Version     		:	20-03-2014
// Creation date	:	20-03-2014
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use the dynamically allocated global memory
//			  so it persists from one kernel call to another
//=============================================================================================

#define BLOCK_SIZE 2
#define NUMBER_OF_BLOCKS 3
#include "stdio.h"

__device__ int* array_global[NUMBER_OF_BLOCKS]; // This time, we keep one array per block in the global memory

__global__ void globalMemoryDynamicPersistentPerBlockAllocate (int Ntot) {
	if (threadIdx.x == 0) {
		array_global[blockIdx.x] = (int*)malloc(blockDim.x * sizeof(int));
	}
	__syncthreads();

	// We should be cheking that malloc didn't return null!
	//if (dataptr[blockIdx.x] == NULL)
	//	return;

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	printf("Block %d, Thread %d [Global id=%d]: got pointer: %p\n", blockIdx.x, threadIdx.x, idx, array_global[blockIdx.x]);

	// Each thread writes its own idx into the global
	array_global[blockIdx.x][threadIdx.x] = idx;
}

// output the value and free
__global__ void globalMemoryDynamicPersistentPerBlockFree (void) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	printf("Block %d, Thread %d [Global id=%d]: assigned value = %d\n",
                      blockIdx.x, threadIdx.x, idx, array_global[blockIdx.x][threadIdx.x]);

	__syncthreads();
	if (threadIdx.x == 0) {
		int* ptr = array_global[blockIdx.x];
		free(ptr);
	}
}

int main() {
	// pointers to host memory
	int *threadIds,*threadOtherIds, *blockIds;
	// pointers to device memory
	int *threadIdsGPU,*threadOtherIdsGPU, *blockIdsGPU;
	// N is the total size that we want
	int N=BLOCK_SIZE*NUMBER_OF_BLOCKS;
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
	dim3 dimGrid ( NUMBER_OF_BLOCKS );

	// Scan information from the threads
	globalMemoryDynamicPersistentPerBlockAllocate<<<dimGrid,dimBlock>>>(N);
	 globalMemoryDynamicPersistentPerBlockFree<<<dimGrid,dimBlock>>>();

	// Copy data from device memory to host memory
	cudaMemcpy(threadIds, threadIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(threadOtherIds, threadOtherIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(blockIds, blockIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);

	// Print all the data about the threads
	if (false) {
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
