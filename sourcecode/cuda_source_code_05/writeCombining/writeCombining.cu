//=============================================================================================
// Name        		: writeCombining.cu
// Author      		: Jose Refojo
// Version     		:	11-02-2013
// Creation date	:	11-02-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will initialize a number of arrays,
//			  then it will grab data from each thread (such as thread position inside the block and block),
//			  save it, send it back into the main memory, and print it
//=============================================================================================

#include "stdio.h"
#include "time.h"
#include <sys/time.h> 

#define NSize 67108864

__global__ void operateInputs(	int *input1GPU,int *input2GPU,int *outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		outputGPU[idx] = input1GPU[idx]+input2GPU[idx];
	}
}


int main() {
	// pointers to host memory
	int *input1,*input2, *output;
	// pointers to device memory
	int *input1GPU,*input2GPU, *outputGPU;

	// pointers to host memory
	int *outputWC;
	// pointers to device memory
	int *outputWCGPU;

	// N is the total size that we want
	int N=NSize;
	int i;

	struct timeval start_total, finish_total;
	struct timeval start_transfer_normal, finish_transfer_normal;
	struct timeval start_transfer_wc, finish_transfer_wc;

	// Allocate arrays input1,input2 and output on host

	int size = N*sizeof(int);

	input1 = (int*) malloc(N*sizeof(int));
	input2 = (int*) malloc(N*sizeof(int));

	output = (int*) malloc(N*sizeof(int));
	cudaHostAlloc((void **)&outputWC, size, cudaHostAllocWriteCombined);


	for (i=0;i<N;i++) {
		input1[i]=-1;
		input2[i]=-2;
		//output[i]=+3;
	}

	// Allocate arrays input1GPU,input2GPU and outputGPU on device
	cudaMalloc ((void **) &input1GPU, sizeof(int)*N);
	cudaMalloc ((void **) &input2GPU, sizeof(int)*N);
	cudaMalloc ((void **) &outputGPU, sizeof(int)*N);
	cudaMalloc ((void **) &outputWCGPU, sizeof(int)*N);

	// Copy data from host memory to device memory
	cudaMemcpy(input1GPU, input1, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(input2GPU, input2, sizeof(int)*N, cudaMemcpyHostToDevice);
	//cudaMemcpy(outputGPU, output, sizeof(int)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=1024;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// *************************************** XYZ test
	gettimeofday(&start_total, NULL);
	double timestart_total=start_total.tv_sec+(start_total.tv_usec/1000000.0);

	// Run the kernels
	operateInputs<<<dimGrid,dimBlock>>>(input1GPU,input2GPU,outputGPU,N);
	operateInputs<<<dimGrid,dimBlock>>>(input1GPU,input2GPU,outputWCGPU,N);

	gettimeofday(&finish_total, NULL);
	double timefinish_total=finish_total.tv_sec+(finish_total.tv_usec/1000000.0);


	gettimeofday(&start_transfer_normal, NULL);
	double time_start_transfer_normal=start_transfer_normal.tv_sec+(start_transfer_normal.tv_usec/1000000.0);
	cudaMemcpy(output, outputGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	gettimeofday(&finish_transfer_normal, NULL);
	double time_finish_transfer_normal=finish_transfer_normal.tv_sec+(finish_transfer_normal.tv_usec/1000000.0);
	printf("transfer of the normal memory took =%f\n",time_finish_transfer_normal-time_start_transfer_normal);

	gettimeofday(&start_transfer_wc, NULL);
	double time_start_transfer_wc=start_transfer_wc.tv_sec+(start_transfer_wc.tv_usec/1000000.0);
	cudaMemcpy(outputWC, outputWCGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	gettimeofday(&finish_transfer_wc, NULL);
	double time_finish_transfer_wc=finish_transfer_wc.tv_sec+(finish_transfer_wc.tv_usec/1000000.0);
	printf("transfer of the wc memory took =%f\n",time_finish_transfer_wc-time_start_transfer_wc);

	printf("which is %e faster\n",(time_finish_transfer_normal-time_start_transfer_normal)/(time_finish_transfer_wc-time_start_transfer_wc));

	// Free the memory
	free(input1);
	free(input2);
	free(output);
//	cudaFreeHost(input1);
//	cudaFreeHost(input2);
	cudaFreeHost(outputWC);

	cudaFree(input1GPU);
	cudaFree(input2GPU);
	cudaFree(outputGPU);
	cudaFree(outputWCGPU);
}
