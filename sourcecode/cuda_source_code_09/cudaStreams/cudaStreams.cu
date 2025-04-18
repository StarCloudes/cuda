//=============================================================================================
// Name        		: cudaStreams.cu
// Author      		: Jose Refojo
// Version     		:	11-04-2018
// Creation date	:	25-03-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use cuda Streams for asynchronous execution
//=============================================================================================

#include "stdio.h"

__global__ void scanThreadInformation0GPU(float *threadIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	float threadIdFloat= (float)(threadIdx.x);
#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanThreadInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		threadIdsGPU[idx]=expf(threadIdFloat)*sinf(threadIdFloat)*cosf(threadIdFloat)*tanf(threadIdFloat);
	}
}
__global__ void scanThreadInformation1GPU(float *threadIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	float threadIdFloat= (float)(threadIdx.x);
#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanThreadInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		threadIdsGPU[idx]=expf(threadIdFloat)*sinf(threadIdFloat)*cosf(threadIdFloat)*tanf(threadIdFloat);
	}
}

__global__ void scanBlockInformation0GPU(float *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	float blockIdFloat= (float)(blockIdx.x);
#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanBlockInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		blockIdsGPU[idx]=acoshf(blockIdFloat)*sinhf(blockIdFloat)*coshf(blockIdFloat)*tanhf(blockIdFloat);
	}
}
__global__ void scanBlockInformation1GPU(float *blockIdsGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	float blockIdFloat= (float)(blockIdx.x);
#ifdef WITH_MY_DEBUG
	printf ("cudaStreams::scanBlockInformationGPU blockIdx.x=%d  threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
#endif
	if ( idx <Ntot ) {
		blockIdsGPU[idx]=atanhf(blockIdFloat)*sinhf(blockIdFloat)*coshf(blockIdFloat)*tanhf(blockIdFloat);
	}
}

bool verbose=false;

int main() {
	int i;

	// pointers to host memory
	float *threadIds, *blockIds;
	// pointers to device memory
	float *threadIds0GPU,*threadIds1GPU,*blockIds0GPU,*blockIds1GPU;
	// N is the total size that we want
	int N=3200000;

	// Declare and allocate two streams
	cudaStream_t stream[4]; 
	for (i = 0; i < 4; ++i) {
		cudaStreamCreate(&stream[i]);
	}

	// Allocate arrays threadIds and blockIds on host
	threadIds = (float*) malloc(N*sizeof(float));
	blockIds = (float*) malloc(N*sizeof(float));

	// Allocate arrays threadIdsGPU and blockIdsGPU on device
	cudaMalloc ((void **) &threadIds0GPU, sizeof(float)*N);
	cudaMalloc ((void **) &threadIds1GPU, sizeof(float)*N);
	cudaMalloc ((void **) &blockIds0GPU , sizeof(float)*N);
	cudaMalloc ((void **) &blockIds1GPU , sizeof(float)*N);
/*
	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(threadIdsGPU, threadIds, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(blockIdsGPU, blockIds, sizeof(int)*N, cudaMemcpyHostToDevice);
*/

	// Compute the execution configuration
	int block_size=128;
	int repetitions=20;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpyAsync(threadIds0GPU, threadIds, sizeof(float)*N, cudaMemcpyHostToDevice,stream[0]);
	for (i = 0; i < repetitions; ++i)
		scanThreadInformation0GPU <<<dimGrid,dimBlock,0,stream[0]>>> (threadIds0GPU, N);

	cudaMemcpyAsync(threadIds1GPU, threadIds, sizeof(float)*N, cudaMemcpyHostToDevice,stream[1]);
	for (i = 0; i < repetitions; ++i)
		scanThreadInformation1GPU <<<dimGrid,dimBlock,0,stream[1]>>> (threadIds1GPU, N);

	cudaMemcpyAsync(blockIds0GPU,   blockIds, sizeof(float)*N, cudaMemcpyHostToDevice,stream[2]);
	for (i = 0; i < repetitions; ++i)
		scanBlockInformation0GPU  <<<dimGrid,dimBlock,0,stream[2]>>> (blockIds0GPU, N);

	cudaMemcpyAsync(blockIds1GPU,   blockIds, sizeof(float)*N, cudaMemcpyHostToDevice,stream[3]);
	for (i = 0; i < repetitions; ++i)
		scanBlockInformation1GPU  <<<dimGrid,dimBlock,0,stream[3]>>> (blockIds1GPU, N);

	cudaMemcpyAsync(threadIds, threadIds0GPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[0]);
	cudaMemcpyAsync(threadIds, threadIds1GPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[1]);
	cudaMemcpyAsync(blockIds , blockIds0GPU , sizeof(float)*N, cudaMemcpyDeviceToHost,stream[2]);
	cudaMemcpyAsync(blockIds , blockIds1GPU , sizeof(float)*N, cudaMemcpyDeviceToHost,stream[3]);

	// Print all the data about the threads

	if (verbose) {
		printf(" dimGrid=%d\n",dimGrid.x);
		for (i=0; i<N; i++) {
		       printf(" threadIds[%d]=%f\n",i,threadIds[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" blockIds[%d]=%f\n",i,blockIds[i]);
		}
	}

	for (int i = 0; i < 4; ++i)
		cudaStreamDestroy(stream[i]); 

	// Free the memory
	free(threadIds);
	free(blockIds); 

	cudaFree(threadIds0GPU);
	cudaFree(threadIds1GPU);
	cudaFree(blockIds0GPU);
	cudaFree(blockIds1GPU);
}
