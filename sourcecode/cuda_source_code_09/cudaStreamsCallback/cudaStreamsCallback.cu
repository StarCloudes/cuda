//=============================================================================================
// Name        		: cudaStreamsCallback.cu
// Author      		: Jose Refojo
// Version     		:	16-02-16
// Creation date	:	16-02-16
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use callbacks with cuda Streams
//=============================================================================================

#include "stdio.h"


void CUDART_CB MyCustomCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback: %s\n", (char *)data);
}

__global__ void scanThreadInformationGPU(int *threadIdsGPU, int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanThreadInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		threadIdsGPU[idx]=threadIdx.x;
	}
}

__global__ void scanBlockInformationGPU(int *threadIdsGPU, int *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;

#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanBlockInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		blockIdsGPU[idx]=blockIdx.x;
	}
}

bool verbose=false;

int main() {
	int i;

	// pointers to host memory
	int *threadIds, *blockIds;
	// pointers to device memory
	int *threadIdsGPU, *blockIdsGPU;
	// N is the total size that we want
	int N=10;

	// Declare and allocate two streams
	cudaStream_t stream[2]; 
	for (i = 0; i < 2; ++i) {
		cudaStreamCreate(&stream[i]);
	}

	// Allocate arrays threadIds and blockIds on host
	threadIds = (int*) malloc(N*sizeof(int));
	blockIds = (int*) malloc(N*sizeof(int));

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIdsGPU, sizeof(int)*N);
	cudaMalloc ((void **) &blockIdsGPU, sizeof(int)*N);
/*
	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice);
*/

	// Copy data from host memory to device memory (not needed, but this is how you do it)
	// Stream 0
	cudaMemcpyAsync(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice,stream[0]);
	cudaStreamAddCallback(stream[0], MyCustomCallback, (void*)("Stream 0: cudaMemcpyAsync - cudaMemcpyHostToDevice has finished\n"),0);

	// Stream 1
	cudaMemcpyAsync(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice,stream[1]);


	// Compute the execution configuration
	int block_size=5;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Scan information from the threads
	scanThreadInformationGPU<<<dimGrid,dimBlock,0,stream[0]>>>(threadIdsGPU, blockIdsGPU, N);
	cudaStreamAddCallback(stream[0], MyCustomCallback, (void*)("Stream 0: scanThreadInformationGPU has finished\n"),0);
	scanBlockInformationGPU<<<dimGrid,dimBlock,0,stream[1]>>>(threadIdsGPU, blockIdsGPU, N);

	// Copy data from device memory to host memory
	cudaMemcpyAsync(threadIds, threadIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost,stream[0]);
	cudaStreamAddCallback(stream[0], MyCustomCallback, (void*)("Stream 0: cudaMemcpyAsync - cudaMemcpyDeviceToHost has finished\n"),0);
	cudaMemcpyAsync(blockIds, blockIdsGPU, sizeof(int)*N, cudaMemcpyDeviceToHost,stream[1]);

	// Print all the data about the threads

	if (verbose) {
		printf(" dimGrid=%d\n",dimGrid.x);
		for (i=0; i<N; i++) {
		       printf(" threadIds[%d]=%d\n",i,threadIds[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" blockIds[%d]=%d\n",i,blockIds[i]);
		}
	}

	for (int i = 0; i < 2; ++i) 
		cudaStreamDestroy(stream[i]); 

	// Free the memory
	free(threadIds);
	free(blockIds); 

	cudaFree(threadIdsGPU);
	cudaFree(blockIdsGPU);
}
