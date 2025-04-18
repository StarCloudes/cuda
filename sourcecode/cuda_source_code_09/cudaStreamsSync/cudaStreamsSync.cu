//=============================================================================================
// Name        		: cudaStreamsSync.cu
// Author      		: Jose Refojo
// Version     		:	28-03-14
// Creation date	:	25-03-13
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use cuda Streams for asynchronous execution
//=============================================================================================

#include "stdio.h"
__global__ void add_arrays_exp_gpu ( float *in1, float *in2, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
       if ( idx <Ntot )
      	 out[idx]=exp(in1[idx])+exp(in2[idx]);
}

bool verbose=false;

int main() {
	int i;

	// pointers to host memory
	float *array1, *array2, *array3;
	float *array12out, *array23out, *array1223out;
	// pointers to device memory
	float *array1GPU, *array2GPU, *array3GPU;
	float *array12outGPU, *array23outGPU, *array1223outGPU;
	// N is the total size that we want
	int N=33554432;
	//int N=1024;
        //134217728

	// Declare and allocate two streams
	cudaStream_t stream[3]; 
	for (i = 0; i < 3; ++i) {
		cudaStreamCreate(&stream[i]);
	}

	// Allocate arrays array1 and array2 on host
	array1 = (float*) malloc(N*sizeof(float));
	array2 = (float*) malloc(N*sizeof(float));
	array3 = (float*) malloc(N*sizeof(float));
	array12out = (float*) malloc(N*sizeof(float));
	array23out = (float*) malloc(N*sizeof(float));
	array1223out = (float*) malloc(N*sizeof(float));

	for (i = 0; i < N; ++i) {
		array1[i]=(float)(i*0.1)+0.1;
		array2[i]=(float)(i*0.1)+0.2;
		array3[i]=(float)(i*0.1)+0.3;
	}
	// Allocate arrays array1GPU and array2GPU on device
	cudaMalloc ((void **) &array1GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array2GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array3GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array12outGPU, sizeof(float)*N);
	cudaMalloc ((void **) &array23outGPU, sizeof(float)*N);
	cudaMalloc ((void **) &array1223outGPU, sizeof(float)*N);

	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpyAsync(array1GPU, array1, sizeof(float)*N, cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(array2GPU, array2, sizeof(float)*N, cudaMemcpyHostToDevice,stream[1]);
	cudaMemcpyAsync(array3GPU, array3, sizeof(float)*N, cudaMemcpyHostToDevice,stream[2]);

	// Compute the execution configuration
	int block_size=1024;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// stream[0] computes 12
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[0]>>>(array1GPU, array2GPU, array12outGPU, N);
	// stream[1] computes 23
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[1]>>>(array2GPU, array3GPU, array23outGPU, N);

	// Make sure that the computations of stream[0] and stream[1] are finished
	// Then start copying back array12out and array23out as soon as possible 
//			cudaEventCreate(&start);
//			cudaEventRecord(start, stream[0]); // We use 0 here because it is the "default" stream
//			cudaStreamWaitEvent(start,stream[0]);

//
//	cudaStreamSynchronize(stream[0]);
//	cudaStreamSynchronize(stream[1]);
//	cudaDeviceSynchronize();
	// stream[2] computes 1223
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[2]>>>(array12outGPU, array23outGPU, array1223outGPU, N);
	cudaMemcpyAsync(array12out, array12outGPU, sizeof(int)*N, cudaMemcpyDeviceToHost,stream[0]);
	cudaMemcpyAsync(array23out, array23outGPU, sizeof(int)*N, cudaMemcpyDeviceToHost,stream[1]);

	// then, stream[2] copies array1223out back
	cudaMemcpyAsync(array1223out, array1223outGPU, sizeof(int)*N, cudaMemcpyDeviceToHost,stream[2]);

	// Print all the data about the threads

	if (verbose) {
		printf(" dimGrid=%d\n",dimGrid.x);
		for (i=0; i<N; i++) {
		       printf(" array12out[%d]=%f\n",i,array12out[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" array23out[%d]=%f\n",i,array23out[i]);
		}
		for (i=0; i<N; i++) {
		       printf(" array1223out[%d]=%f\n",i,array1223out[i]);
		}
	}

	for (int i = 0; i < 3; ++i) 
		cudaStreamDestroy(stream[i]); 

	// Free the memory
	free(array1);
	free(array2); 
	free(array3);
	free(array12out);
	free(array23out); 
	free(array1223out);

	cudaFree(array1GPU);
	cudaFree(array2GPU);
	cudaFree(array3GPU);
	cudaFree(array12outGPU);
	cudaFree(array23outGPU);
	cudaFree(array1223outGPU);
}
