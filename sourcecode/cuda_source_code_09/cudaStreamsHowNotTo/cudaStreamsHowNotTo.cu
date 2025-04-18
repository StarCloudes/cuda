//=============================================================================================
// Name        		: cudaStreamsHowNotTo.cu
// Author      		: Jose Refojo
// Version     		:	02-05-2018
// Creation date	:	25-03-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how NOT to use cuda Streams for asynchronous execution
//=============================================================================================

#include "stdio.h"
__global__ void add_arrays_exp_gpu ( float *in1, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
       if ( idx <Ntot )
      	 out[idx]+=exp(in1[idx]);
}

bool verbose=true;

int main() {
	int i;

	// pointers to host memory
	float *array0,*array1, *array2, *array3;
	float *array0out,*array1out, *array2out, *array3out;
	// pointers to device memory
	float *array0GPU,*array1GPU, *array2GPU, *array3GPU;
	float *array0outGPU,*array1outGPU, *array2outGPU, *array3outGPU;
	// N is the total size that we want
	//int N=13554432;
	//int N=1024;
	int N=4194304;
	//int N=10;
        //134217728

	// Compute the execution configuration
	int block_size=1024;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );


	// Allocate arrays array1 and array2 on host
	array0 = (float*) malloc(N*sizeof(float));
	array1 = (float*) malloc(N*sizeof(float));
	array2 = (float*) malloc(N*sizeof(float));
	array3 = (float*) malloc(N*sizeof(float));
	array0out = (float*) malloc(N*sizeof(float));
	array1out = (float*) malloc(N*sizeof(float));
	array2out = (float*) malloc(N*sizeof(float));
	array3out = (float*) malloc(N*sizeof(float));

	for (i = 0; i < N; ++i) {
		array0[i]=(float)(i*0.01)/(float)(N);
		array1[i]=(float)(i*0.10)/(float)(N);
		array2[i]=(float)(i*0.20)/(float)(N);
		array3[i]=(float)(i*0.30)/(float)(N);
	}
	// Allocate arrays array1GPU and array2GPU on device
	cudaMalloc ((void **) &array0GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array1GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array2GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array3GPU, sizeof(float)*N);
	cudaMalloc ((void **) &array0outGPU, sizeof(float)*N);
	cudaMalloc ((void **) &array1outGPU, sizeof(float)*N);
	cudaMalloc ((void **) &array2outGPU, sizeof(float)*N);
	cudaMalloc ((void **) &array3outGPU, sizeof(float)*N);

	// Declare and allocate two streams
	cudaStream_t stream[4]; 
	for (i = 0; i < 4; ++i) {
		cudaStreamCreate(&stream[i]);
	}

	//////////////////////////////////////////////////////////////////
	// THIS IS HOW TO USE CUDA STREAMS PROPERLY:

	// Copy data from host memory to device memory (not needed, but this is how you do it)
	// stream[0] copies array0 to device
	cudaMemcpyAsync(array0GPU, array0, sizeof(float)*N, cudaMemcpyHostToDevice,stream[0]);
	// stream[0] computes array0
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[0]>>>(array0GPU, array0outGPU, N);

	// stream[1] copies array1
	cudaMemcpyAsync(array1GPU, array1, sizeof(float)*N, cudaMemcpyHostToDevice,stream[1]);
	// stream[1] computes array1
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[1]>>>(array1GPU, array1outGPU, N);

	// stream[2] copies array2
	cudaMemcpyAsync(array2GPU, array2, sizeof(float)*N, cudaMemcpyHostToDevice,stream[2]);
	// stream[2] computes array2
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[2]>>>(array2GPU, array2outGPU, N);

	// stream[3] copies array3
	cudaMemcpyAsync(array3GPU, array3, sizeof(float)*N, cudaMemcpyHostToDevice,stream[3]);
	// stream[2] computes array3
	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[3]>>>(array3GPU, array3outGPU, N);

	// stream[0] copies array0 to host
	cudaMemcpyAsync(array0out, array0outGPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[0]);
	// stream[1] copies array1 to host
	cudaMemcpyAsync(array1out, array1outGPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[1]);
	// stream[2] copies array2 to host
	cudaMemcpyAsync(array2out, array2outGPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[2]);
	// stream[3] copies array3 to host
	cudaMemcpyAsync(array3out, array3outGPU, sizeof(float)*N, cudaMemcpyDeviceToHost,stream[3]);


	//////////////////////////////////////////////////////////////////
	// What if we tried to use loops instead of unrolling the loops ourselves?
	float *array[4],*arrayOut[4],*arrayGPU[4],*arrayOutGPU[4];
	array[0]=array0;	arrayOut[0]=array0out;	arrayGPU[0]=array0GPU;	arrayOutGPU[0]=array0outGPU;
	array[1]=array1;	arrayOut[1]=array1out;	arrayGPU[1]=array1GPU;	arrayOutGPU[1]=array1outGPU;
	array[2]=array2;	arrayOut[2]=array2out;	arrayGPU[2]=array2GPU;	arrayOutGPU[2]=array2outGPU;
	array[3]=array3;	arrayOut[3]=array3out;	arrayGPU[3]=array3GPU;	arrayOutGPU[3]=array3outGPU;
	// This worrks:
	for (i = 0; i < 4; ++i)	cudaMemcpyAsync(arrayGPU[i], array[i], sizeof(float)*N, cudaMemcpyHostToDevice,stream[i]); 			// stream[i] copies array[i] to device
	for (i = 0; i < 4; ++i)	add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[i]>>>(arrayGPU[i], arrayOutGPU[i], N);				// stream[i] computes array[i]
	for (i = 0; i < 4; ++i)	cudaMemcpyAsync(arrayOut[i], arrayOutGPU[i], sizeof(float)*N, cudaMemcpyDeviceToHost,stream[i]);	// stream[i] copies array[i] to host
	// But this doesn't:
	//////////////////////////////////////////////////////////////////
	// THIS IS HOW *NOT* TO USE CUDA STREAMS PROPERLY - IF YOU DO THIS, THE CALLS WILL BE SERIALIZED:
	for (i = 0; i < 4; ++i) {
		// stream[i] copies array[i] to device
		cudaMemcpyAsync(arrayGPU[i], array[i], sizeof(float)*N, cudaMemcpyHostToDevice,stream[i]);
		// stream[i] computes array[i]
		add_arrays_exp_gpu<<<dimGrid,dimBlock,0,stream[i]>>>(arrayGPU[i], arrayOutGPU[i], N);
		// stream[i] copies array[i] to host
		cudaMemcpyAsync(arrayOut[i], arrayOutGPU[i], sizeof(float)*N, cudaMemcpyDeviceToHost,stream[i]);
	}




	cudaDeviceSynchronize();

	// Print all the data about the threads
	int printLimit=min(10,N);
	if (verbose) {
		printf(" dimGrid=%d\n",dimGrid.x);
		for (i=0; i<printLimit; i++) {
		       printf(" array0out[%d]=%f array1out[%d]=%f array2out[%d]=%f array3out[%d]=%f\n",i,array0out[i],i,array1out[i],i,array2out[i],i,array3out[i]);
		}
	}

	for (int i = 0; i < 4; ++i) 
		cudaStreamDestroy(stream[i]); 

	// Free the memory 
	free(array0);
	free(array1);
	free(array2); 
	free(array3);
	free(array0out);
	free(array1out);
	free(array2out); 
	free(array3out);

	cudaFree(array0GPU);
	cudaFree(array1GPU);
	cudaFree(array2GPU);
	cudaFree(array3GPU);
	cudaFree(array0outGPU);
	cudaFree(array1outGPU);
	cudaFree(array2outGPU);
	cudaFree(array3outGPU);
}
