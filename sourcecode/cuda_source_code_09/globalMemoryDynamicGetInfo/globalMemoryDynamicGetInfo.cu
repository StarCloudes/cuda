//=============================================================================================
// Name        		: globalMemoryDynamicGetInfo.cu
// Author      		: Jose Refojo
// Version     		:	07-03-2016
// Creation date	:	07-03-2016
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use the dynamically allocated global memory
//			  so it persists from one kernel call to another
//=============================================================================================

#define BLOCK_SIZE 1024
#define NUMBER_OF_BLOCKS 256
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
	//printf("Block %d, Thread %d [Global id=%d]: got pointer: %p\n", blockIdx.x, threadIdx.x, idx, array_global[blockIdx.x]);

	// Each thread writes its own idx into the global
	array_global[blockIdx.x][threadIdx.x] = idx;
}

// output the value and free
__global__ void globalMemoryDynamicPersistentPerBlockFree (void) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	//printf("Block %d, Thread %d [Global id=%d]: assigned value = %d\n",
          //            blockIdx.x, threadIdx.x, idx, array_global[blockIdx.x][threadIdx.x]);

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

	size_t freeGlobalMemory,totalGlobalMemory;
	cudaMemGetInfo (&freeGlobalMemory,&totalGlobalMemory);
	printf ("[0] freeGlobalMemory= %lu bytes\t",freeGlobalMemory);
	printf ("totalGlobalMemory= %lu bytes\n",totalGlobalMemory);

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIdsGPU, 		sizeof(int)*N);
	cudaMalloc ((void **) &threadOtherIdsGPU,	sizeof(int)*N);
	cudaMalloc ((void **) &blockIdsGPU, 		sizeof(int)*N);

	cudaMemGetInfo (&freeGlobalMemory,&totalGlobalMemory);
	printf ("[1] freeGlobalMemory= %lu bytes\t",freeGlobalMemory);
	printf ("totalGlobalMemory= %lu bytes\n",totalGlobalMemory);

	size_t globalMemoryLimitSize;
	cudaDeviceGetLimit(&globalMemoryLimitSize, cudaLimitMallocHeapSize);
	printf ("The default limit for the malloc heap size is %lu bytes\n",globalMemoryLimitSize);
	// We want more than 8388608 bytes?
	size_t desiredGlobalMemoryLimitSize = 10000000;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, desiredGlobalMemoryLimitSize);

	cudaDeviceGetLimit(&globalMemoryLimitSize, cudaLimitMallocHeapSize);
	printf ("The new limit for the malloc heap size is %lu bytes\n",globalMemoryLimitSize);

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( NUMBER_OF_BLOCKS );

	// Scan information from the threads
	globalMemoryDynamicPersistentPerBlockAllocate<<<dimGrid,dimBlock>>>(N);
	globalMemoryDynamicPersistentPerBlockFree<<<dimGrid,dimBlock>>>();


	cudaMemGetInfo (&freeGlobalMemory,&totalGlobalMemory);
	printf ("[2] freeGlobalMemory= %lu bytes\t",freeGlobalMemory);
	printf ("totalGlobalMemory= %lu bytes\n",totalGlobalMemory);

	// Free the memory
	free(threadIds);
	free(threadOtherIds);
	free(blockIds); 

	cudaFree(threadIdsGPU);
	cudaFree(threadOtherIdsGPU);
	cudaFree(blockIdsGPU);
}
