//=============================================================================================
// Name        		: align.cu
// Author      		: Jose Refojo
// Creation date	:	18-10-2010
// Version     		:	14-10-2019
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will perform a series of memory transfers between aligned and unaligned memory of different word sizes, as print out the timings
//=============================================================================================

#include "stdio.h"
#include "time.h"
#include <unistd.h>
#include <sys/time.h> 

int gpuToUse;

#define NSize 512;

typedef struct __align__ (4) {
	int x;
} X_ALIGNED;

typedef struct {
	int x;
} X_NOT_ALIGNED;

typedef struct __align__ (8) {
	int x;
	int y;
} XY_ALIGNED;

typedef struct {
	int x;
	int y;
} XY_NOT_ALIGNED;

typedef struct __align__ (16) {
	int x;
	int y;
	int z;
} XYZ_ALIGNED;

typedef struct {
	int x;
	int y;
	int z;
} XYZ_NOT_ALIGNED;

__global__ void operateXInputsAlignedGPU(	X_ALIGNED *X_ALIGNED_input1GPU,
						X_ALIGNED *X_ALIGNED_input2GPU,
						X_ALIGNED *X_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		X_ALIGNED_outputGPU[idx].x = X_ALIGNED_input1GPU[idx].x+X_ALIGNED_input2GPU[idx].x;
	}
}
__global__ void operateXInputsNotAlignedGPU(	X_NOT_ALIGNED *X_NOT_ALIGNED_input1GPU,
						X_NOT_ALIGNED *X_NOT_ALIGNED_input2GPU,
						X_NOT_ALIGNED *X_NOT_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		X_NOT_ALIGNED_outputGPU[idx].x = X_NOT_ALIGNED_input1GPU[idx].x+X_NOT_ALIGNED_input2GPU[idx].x;
	}
}

__global__ void operateXYInputsAlignedGPU(	XY_ALIGNED *XY_ALIGNED_input1GPU,
						XY_ALIGNED *XY_ALIGNED_input2GPU,
						XY_ALIGNED *XY_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		XY_ALIGNED_outputGPU[idx].x = XY_ALIGNED_input1GPU[idx].x+XY_ALIGNED_input2GPU[idx].x;
		XY_ALIGNED_outputGPU[idx].y = XY_ALIGNED_input1GPU[idx].y+XY_ALIGNED_input2GPU[idx].y;
	}
}
__global__ void operateXYInputsNotAlignedGPU(	XY_NOT_ALIGNED *XY_NOT_ALIGNED_input1GPU,
						XY_NOT_ALIGNED *XY_NOT_ALIGNED_input2GPU,
						XY_NOT_ALIGNED *XY_NOT_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		XY_NOT_ALIGNED_outputGPU[idx].x = XY_NOT_ALIGNED_input1GPU[idx].x+XY_NOT_ALIGNED_input2GPU[idx].x;
		XY_NOT_ALIGNED_outputGPU[idx].y = XY_NOT_ALIGNED_input1GPU[idx].y+XY_NOT_ALIGNED_input2GPU[idx].y;
	}
}

__global__ void operateXYZInputsAlignedGPU(	XYZ_ALIGNED *XYZ_ALIGNED_input1GPU,
						XYZ_ALIGNED *XYZ_ALIGNED_input2GPU,
						XYZ_ALIGNED *XYZ_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		XYZ_ALIGNED_outputGPU[idx].x = XYZ_ALIGNED_input1GPU[idx].x+XYZ_ALIGNED_input2GPU[idx].x;
		XYZ_ALIGNED_outputGPU[idx].y = XYZ_ALIGNED_input1GPU[idx].y+XYZ_ALIGNED_input2GPU[idx].y;
		XYZ_ALIGNED_outputGPU[idx].z = XYZ_ALIGNED_input1GPU[idx].z+XYZ_ALIGNED_input2GPU[idx].z;
	}
}
__global__ void operateXYZInputsNotAlignedGPU(	XYZ_NOT_ALIGNED *XYZ_NOT_ALIGNED_input1GPU,
						XYZ_NOT_ALIGNED *XYZ_NOT_ALIGNED_input2GPU,
						XYZ_NOT_ALIGNED *XYZ_NOT_ALIGNED_outputGPU,int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx <Ntot ) {
		XYZ_NOT_ALIGNED_outputGPU[idx].x = XYZ_NOT_ALIGNED_input1GPU[idx].x+XYZ_NOT_ALIGNED_input2GPU[idx].x;
		XYZ_NOT_ALIGNED_outputGPU[idx].y = XYZ_NOT_ALIGNED_input1GPU[idx].y+XYZ_NOT_ALIGNED_input2GPU[idx].y;
		XYZ_NOT_ALIGNED_outputGPU[idx].z = XYZ_NOT_ALIGNED_input1GPU[idx].z+XYZ_NOT_ALIGNED_input2GPU[idx].z;
	}
}


int parseArguments (int argc, char *argv[]) {
	int c,tmp;
	int numberOfCards;
	while ((c = getopt (argc, argv, "g:")) != -1) {
		printf("c=:%d\n",c);
		switch(c) {
			case 'g':
				tmp = atoi(optarg);
				if ( cudaGetDeviceCount(&numberOfCards)!=cudaSuccess ) {
					printf("No CUDA-enabled devices were found\n");
					return -1;
				}
				if (tmp<numberOfCards) {
					gpuToUse=tmp;
					struct cudaDeviceProp x;
					cudaGetDeviceProperties(&x, tmp);
					printf("GPU model name: %s\n",x.name);
				}
				break;
			default:
				fprintf(stderr, "Invalid option given\n");
				//printUsage();
				return -1;
		}
	}
	return 0;
}

int main(int argc, char *argv[]) {
	gpuToUse=0;

	// pointers to host memory
	int *input1,*input2, *output;
	// pointers to device memory
	int *input1GPU,*input2GPU, *outputGPU;

	parseArguments(argc, argv);

	cudaSetDevice(gpuToUse);

	printf("size of (X_ALIGNED)    =%lu\n",sizeof(X_ALIGNED));
	printf("size of (X_NOT_ALIGNED)=%lu\n",sizeof(X_NOT_ALIGNED));
	printf("size of (XY_ALIGNED)    =%lu\n",sizeof(XY_ALIGNED));
	printf("size of (XY_NOT_ALIGNED)=%lu\n",sizeof(XY_NOT_ALIGNED));
	printf("size of (XYZ_ALIGNED)    =%lu\n",sizeof(XYZ_ALIGNED));
	printf("size of (XYZ_NOT_ALIGNED)=%lu\n",sizeof(XYZ_NOT_ALIGNED));

	// pointers to host memory
	X_ALIGNED *X_ALIGNED_input1,*X_ALIGNED_input2, *X_ALIGNED_output;
	X_NOT_ALIGNED *X_NOT_ALIGNED_input1,*X_NOT_ALIGNED_input2, *X_NOT_ALIGNED_output;
	// pointers to device memory
	X_ALIGNED *X_ALIGNED_input1GPU,*X_ALIGNED_input2GPU, *X_ALIGNED_outputGPU;
	X_NOT_ALIGNED *X_NOT_ALIGNED_input1GPU,*X_NOT_ALIGNED_input2GPU, *X_NOT_ALIGNED_outputGPU;

	// pointers to host memory
	XY_ALIGNED *XY_ALIGNED_input1,*XY_ALIGNED_input2, *XY_ALIGNED_output;
	XY_NOT_ALIGNED *XY_NOT_ALIGNED_input1,*XY_NOT_ALIGNED_input2, *XY_NOT_ALIGNED_output;
	// pointers to device memory
	XY_ALIGNED *XY_ALIGNED_input1GPU,*XY_ALIGNED_input2GPU, *XY_ALIGNED_outputGPU;
	XY_NOT_ALIGNED *XY_NOT_ALIGNED_input1GPU,*XY_NOT_ALIGNED_input2GPU, *XY_NOT_ALIGNED_outputGPU;

	// pointers to host memory
	XYZ_ALIGNED *XYZ_ALIGNED_input1,*XYZ_ALIGNED_input2, *XYZ_ALIGNED_output;
	XYZ_NOT_ALIGNED *XYZ_NOT_ALIGNED_input1,*XYZ_NOT_ALIGNED_input2, *XYZ_NOT_ALIGNED_output;
	// pointers to device memory
	XYZ_ALIGNED *XYZ_ALIGNED_input1GPU,*XYZ_ALIGNED_input2GPU, *XYZ_ALIGNED_outputGPU;
	XYZ_NOT_ALIGNED *XYZ_NOT_ALIGNED_input1GPU,*XYZ_NOT_ALIGNED_input2GPU, *XYZ_NOT_ALIGNED_outputGPU;

	// N is the total size that we want
	int N=NSize;
	int numberOfTimes = 10000000;
	int i;

	struct timeval start_X_1, finish_X_1;
	struct timeval start_X_2, finish_X_2;
	struct timeval start_XY_1, finish_XY_1;
	struct timeval start_XY_2, finish_XY_2;
	struct timeval start_XYZ_1, finish_XYZ_1;
	struct timeval start_XYZ_2, finish_XYZ_2;

	// Allocate arrays input1,input2 and output on host
	input1 = (int*) malloc(N*sizeof(int));
	input2 = (int*) malloc(N*sizeof(int));
	output = (int*) malloc(N*sizeof(int));


	// For the X_ALIGNED test
	X_ALIGNED_input1 = (X_ALIGNED*) malloc(N*sizeof(X_ALIGNED));
	X_ALIGNED_input2 = (X_ALIGNED*) malloc(N*sizeof(X_ALIGNED));
	X_ALIGNED_output = (X_ALIGNED*) malloc(N*sizeof(X_ALIGNED));
	// For the X_NOT_ALIGNED test
	X_NOT_ALIGNED_input1 = (X_NOT_ALIGNED*) malloc(N*sizeof(X_NOT_ALIGNED));
	X_NOT_ALIGNED_input2 = (X_NOT_ALIGNED*) malloc(N*sizeof(X_NOT_ALIGNED));
	X_NOT_ALIGNED_output = (X_NOT_ALIGNED*) malloc(N*sizeof(X_NOT_ALIGNED));

	// For the XY_ALIGNED test
	XY_ALIGNED_input1 = (XY_ALIGNED*) malloc(N*sizeof(XY_ALIGNED));
	XY_ALIGNED_input2 = (XY_ALIGNED*) malloc(N*sizeof(XY_ALIGNED));
	XY_ALIGNED_output = (XY_ALIGNED*) malloc(N*sizeof(XY_ALIGNED));
	// For the XY_NOT_ALIGNED test
	XY_NOT_ALIGNED_input1 = (XY_NOT_ALIGNED*) malloc(N*sizeof(XY_NOT_ALIGNED));
	XY_NOT_ALIGNED_input2 = (XY_NOT_ALIGNED*) malloc(N*sizeof(XY_NOT_ALIGNED));
	XY_NOT_ALIGNED_output = (XY_NOT_ALIGNED*) malloc(N*sizeof(XY_NOT_ALIGNED));

	// For the XYZ_ALIGNED test
	XYZ_ALIGNED_input1 = (XYZ_ALIGNED*) malloc(N*sizeof(XYZ_ALIGNED));
	XYZ_ALIGNED_input2 = (XYZ_ALIGNED*) malloc(N*sizeof(XYZ_ALIGNED));
	XYZ_ALIGNED_output = (XYZ_ALIGNED*) malloc(N*sizeof(XYZ_ALIGNED));
	// For the XYZ_NOT_ALIGNED test
	XYZ_NOT_ALIGNED_input1 = (XYZ_NOT_ALIGNED*) malloc(N*sizeof(XYZ_NOT_ALIGNED));
	XYZ_NOT_ALIGNED_input2 = (XYZ_NOT_ALIGNED*) malloc(N*sizeof(XYZ_NOT_ALIGNED));
	XYZ_NOT_ALIGNED_output = (XYZ_NOT_ALIGNED*) malloc(N*sizeof(XYZ_NOT_ALIGNED));

	for (i=0;i<N;i++) {
		input1[i]=-1;
		input2[i]=-2;
		output[i]=+3;

		X_ALIGNED_input1[i].x=-1;
		X_NOT_ALIGNED_input1[i].x=-1;

		X_ALIGNED_input2[i].x=-1;
		X_NOT_ALIGNED_input2[i].x=-1;

		X_ALIGNED_output[i].x=0;
		X_NOT_ALIGNED_output[i].x=0;


		XY_ALIGNED_input1[i].x=-1;		XY_ALIGNED_input1[i].y=-2;	
		XY_NOT_ALIGNED_input1[i].x=-1;		XY_NOT_ALIGNED_input1[i].y=-2;	

		XY_ALIGNED_input2[i].x=-1;		XY_ALIGNED_input2[i].y=-2;	
		XY_NOT_ALIGNED_input2[i].x=-1;		XY_NOT_ALIGNED_input2[i].y=-2;	

		XY_ALIGNED_output[i].x=0;		XY_ALIGNED_output[i].y=0;	
		XY_NOT_ALIGNED_output[i].x=0;		XY_NOT_ALIGNED_output[i].y=0;	



		XYZ_ALIGNED_input1[i].x=-1;		XYZ_ALIGNED_input1[i].y=-2;		XYZ_ALIGNED_input1[i].z=-3;
		XYZ_NOT_ALIGNED_input1[i].x=-1;		XYZ_NOT_ALIGNED_input1[i].y=-2;		XYZ_NOT_ALIGNED_input1[i].z=-3;

		XYZ_ALIGNED_input2[i].x=-1;		XYZ_ALIGNED_input2[i].y=-2;		XYZ_ALIGNED_input2[i].z=-3;
		XYZ_NOT_ALIGNED_input2[i].x=-1;		XYZ_NOT_ALIGNED_input2[i].y=-2;		XYZ_NOT_ALIGNED_input2[i].z=-3;

		XYZ_ALIGNED_output[i].x=0;		XYZ_ALIGNED_output[i].y=0;		XYZ_ALIGNED_output[i].z=0;
		XYZ_NOT_ALIGNED_output[i].x=0;		XYZ_NOT_ALIGNED_output[i].y=0;		XYZ_NOT_ALIGNED_output[i].z=0;
	}

	// Allocate arrays input1GPU,input2GPU and outputGPU on device
	cudaMalloc ((void **) &input1GPU, sizeof(int)*N);
	cudaMalloc ((void **) &input2GPU, sizeof(int)*N);
	cudaMalloc ((void **) &outputGPU, sizeof(int)*N);

	// For the X_ALIGNED test
	cudaMalloc ((void **) &X_ALIGNED_input1GPU, sizeof(X_ALIGNED)*N);
	cudaMalloc ((void **) &X_ALIGNED_input2GPU, sizeof(X_ALIGNED)*N);
	cudaMalloc ((void **) &X_ALIGNED_outputGPU, sizeof(X_ALIGNED)*N);
	// For the X_NOT_ALIGNED test
	cudaMalloc ((void **) &X_NOT_ALIGNED_input1GPU, sizeof(X_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &X_NOT_ALIGNED_input2GPU, sizeof(X_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &X_NOT_ALIGNED_outputGPU, sizeof(X_NOT_ALIGNED)*N);

	// For the XY_ALIGNED test
	cudaMalloc ((void **) &XY_ALIGNED_input1GPU, sizeof(XY_ALIGNED)*N);
	cudaMalloc ((void **) &XY_ALIGNED_input2GPU, sizeof(XY_ALIGNED)*N);
	cudaMalloc ((void **) &XY_ALIGNED_outputGPU, sizeof(XY_ALIGNED)*N);
	// For the XY_NOT_ALIGNED test
	cudaMalloc ((void **) &XY_NOT_ALIGNED_input1GPU, sizeof(XY_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &XY_NOT_ALIGNED_input2GPU, sizeof(XY_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &XY_NOT_ALIGNED_outputGPU, sizeof(XY_NOT_ALIGNED)*N);

	// For the XYZ_ALIGNED test
	cudaMalloc ((void **) &XYZ_ALIGNED_input1GPU, sizeof(XYZ_ALIGNED)*N);
	cudaMalloc ((void **) &XYZ_ALIGNED_input2GPU, sizeof(XYZ_ALIGNED)*N);
	cudaMalloc ((void **) &XYZ_ALIGNED_outputGPU, sizeof(XYZ_ALIGNED)*N);
	// For the XYZ_NOT_ALIGNED test
	cudaMalloc ((void **) &XYZ_NOT_ALIGNED_input1GPU, sizeof(XYZ_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &XYZ_NOT_ALIGNED_input2GPU, sizeof(XYZ_NOT_ALIGNED)*N);
	cudaMalloc ((void **) &XYZ_NOT_ALIGNED_outputGPU, sizeof(XYZ_NOT_ALIGNED)*N);

	// Copy data from host memory to device memory
	cudaMemcpy(input1GPU, input1, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(input2GPU, input2, sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outputGPU, output, sizeof(int)*N, cudaMemcpyHostToDevice);


	// For the X_ALIGNED test
	cudaMemcpy(X_ALIGNED_input1GPU, X_ALIGNED_input1, sizeof(X_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(X_ALIGNED_input2GPU, X_ALIGNED_input2, sizeof(X_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(X_ALIGNED_outputGPU, X_ALIGNED_output, sizeof(X_ALIGNED)*N, cudaMemcpyHostToDevice);
	// For the X_NOT_ALIGNED test
	cudaMemcpy(X_NOT_ALIGNED_input1GPU, X_NOT_ALIGNED_input1, sizeof(X_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(X_NOT_ALIGNED_input2GPU, X_NOT_ALIGNED_input2, sizeof(X_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(X_NOT_ALIGNED_outputGPU, X_NOT_ALIGNED_output, sizeof(X_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);

	// For the XY_ALIGNED test
	cudaMemcpy(XY_ALIGNED_input1GPU, XY_ALIGNED_input1, sizeof(XY_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XY_ALIGNED_input2GPU, XY_ALIGNED_input2, sizeof(XY_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XY_ALIGNED_outputGPU, XY_ALIGNED_output, sizeof(XY_ALIGNED)*N, cudaMemcpyHostToDevice);
	// For the XY_NOT_ALIGNED test
	cudaMemcpy(XY_NOT_ALIGNED_input1GPU, XY_NOT_ALIGNED_input1, sizeof(XY_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XY_NOT_ALIGNED_input2GPU, XY_NOT_ALIGNED_input2, sizeof(XY_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XY_NOT_ALIGNED_outputGPU, XY_NOT_ALIGNED_output, sizeof(XY_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);

	// For the XYZ_ALIGNED test
	cudaMemcpy(XYZ_ALIGNED_input1GPU, XYZ_ALIGNED_input1, sizeof(XYZ_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XYZ_ALIGNED_input2GPU, XYZ_ALIGNED_input2, sizeof(XYZ_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XYZ_ALIGNED_outputGPU, XYZ_ALIGNED_output, sizeof(XYZ_ALIGNED)*N, cudaMemcpyHostToDevice);
	// For the XYZ_NOT_ALIGNED test
	cudaMemcpy(XYZ_NOT_ALIGNED_input1GPU, XYZ_NOT_ALIGNED_input1, sizeof(XYZ_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XYZ_NOT_ALIGNED_input2GPU, XYZ_NOT_ALIGNED_input2, sizeof(XYZ_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(XYZ_NOT_ALIGNED_outputGPU, XYZ_NOT_ALIGNED_output, sizeof(XYZ_NOT_ALIGNED)*N, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=64;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// *************************************** XYZ test
	gettimeofday(&start_XYZ_1, NULL);
	double timeStart_XYZ_1=start_XYZ_1.tv_sec+(start_XYZ_1.tv_usec/1000000.0);

	// Scan information from the threads
	for (i=0; i<numberOfTimes; i++) {
		operateXYZInputsNotAlignedGPU<<<dimGrid,dimBlock>>>(XYZ_NOT_ALIGNED_input1GPU,XYZ_NOT_ALIGNED_input2GPU,XYZ_NOT_ALIGNED_outputGPU,N);
	}

	gettimeofday(&finish_XYZ_1, NULL);
	double timeFinish_XYZ_1=finish_XYZ_1.tv_sec+(finish_XYZ_1.tv_usec/1000000.0);

	gettimeofday(&start_XYZ_2, NULL);
	// Set up aligned test

	double timeStart_XYZ_2=start_XYZ_2.tv_sec+(start_XYZ_2.tv_usec/1000000.0);

	for (i=0; i<numberOfTimes; i++) {
		operateXYZInputsAlignedGPU<<<dimGrid,dimBlock>>>(XYZ_ALIGNED_input1GPU,XYZ_ALIGNED_input2GPU,XYZ_ALIGNED_outputGPU,N);
	}
	// Copy data from device memory to host memory

	gettimeofday(&finish_XYZ_2, NULL);
	double timeFinish_XYZ_2=finish_XYZ_2.tv_sec+(finish_XYZ_2.tv_usec/1000000.0);

	cudaMemcpy(input1, input1GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(input2, input2GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	printf("for the XYZ unaligned test, timeFinish_XYZ_1-timeStart_XYZ_1 =%e\n",timeFinish_XYZ_1-timeStart_XYZ_1);
	printf("for the XYZ aligned test, timeFinish_XYZ_2-timeStart_XYZ_2 =%e\n",timeFinish_XYZ_2-timeStart_XYZ_2);
	printf("which is %e faster\n",(timeFinish_XYZ_1-timeStart_XYZ_1)/(timeFinish_XYZ_2-timeStart_XYZ_2));


	// *************************************** XY test
	gettimeofday(&start_XY_1, NULL);
	double timeStart_XY_1=start_XY_1.tv_sec+(start_XY_1.tv_usec/1000000.0);

	// Scan information from the threads
	for (i=0; i<numberOfTimes; i++) {
		operateXYInputsNotAlignedGPU<<<dimGrid,dimBlock>>>(XY_NOT_ALIGNED_input1GPU,XY_NOT_ALIGNED_input2GPU,XY_NOT_ALIGNED_outputGPU,N);
	}

	gettimeofday(&finish_XY_1, NULL);
	double timeFinish_XY_1=finish_XY_1.tv_sec+(finish_XY_1.tv_usec/1000000.0);

	gettimeofday(&start_XY_2, NULL);
	// Set up aligned test

	double timeStart_XY_2=start_XY_2.tv_sec+(start_XY_2.tv_usec/1000000.0);

	for (i=0; i<numberOfTimes; i++) {
		operateXYInputsAlignedGPU<<<dimGrid,dimBlock>>>(XY_ALIGNED_input1GPU,XY_ALIGNED_input2GPU,XY_ALIGNED_outputGPU,N);
	}
	// Copy data from device memory to host memory

	gettimeofday(&finish_XY_2, NULL);
	double timeFinish_XY_2=finish_XY_2.tv_sec+(finish_XY_2.tv_usec/1000000.0);

	cudaMemcpy(input1, input1GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(input2, input2GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	printf("for the XY unaligned test, timeFinish_XY_1-timeStart_XY_1 =%e\n",timeFinish_XY_1-timeStart_XY_1);
	printf("for the XY aligned test, timeFinish_XY_2-timeStart_XY_2 =%e\n",timeFinish_XY_2-timeStart_XY_2);
	printf("which is %e faster\n",(timeFinish_XY_1-timeStart_XY_1)/(timeFinish_XY_2-timeStart_XY_2));

	// *************************************** X test
	gettimeofday(&start_X_1, NULL);
	double timeStart_X_1=start_X_1.tv_sec+(start_X_1.tv_usec/1000000.0);

	// Scan information from the threads
	for (i=0; i<numberOfTimes; i++) {
		operateXInputsNotAlignedGPU<<<dimGrid,dimBlock>>>(X_NOT_ALIGNED_input1GPU,X_NOT_ALIGNED_input2GPU,X_NOT_ALIGNED_outputGPU,N);
	}

	gettimeofday(&finish_X_1, NULL);
	double timeFinish_X_1=finish_X_1.tv_sec+(finish_X_1.tv_usec/1000000.0);

	gettimeofday(&start_X_2, NULL);
	// Set up aligned test

	double timeStart_X_2=start_X_2.tv_sec+(start_X_2.tv_usec/1000000.0);

	for (i=0; i<numberOfTimes; i++) {
		operateXInputsAlignedGPU<<<dimGrid,dimBlock>>>(X_ALIGNED_input1GPU,X_ALIGNED_input2GPU,X_ALIGNED_outputGPU,N);
	}
	// Copy data from device memory to host memory

	gettimeofday(&finish_X_2, NULL);
	double timeFinish_X_2=finish_X_2.tv_sec+(finish_X_2.tv_usec/1000000.0);

	cudaMemcpy(input1, input1GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(input2, input2GPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, outputGPU, sizeof(int)*N, cudaMemcpyDeviceToHost);
	
	printf("for the X unaligned test, timeFinish_X_1-timeStart_X_1 =%e\n",timeFinish_X_1-timeStart_X_1);
	printf("for the X aligned test, timeFinish_X_2-timeStart_X_2 =%e\n",timeFinish_X_2-timeStart_X_2);
	printf("which is %e faster\n",(timeFinish_X_1-timeStart_X_1)/(timeFinish_X_2-timeStart_X_2));

	// Free the memory
	free(input1);
	free(input2);
	free(output);

	free(XYZ_ALIGNED_input1);
	free(XYZ_ALIGNED_input2);
	free(XYZ_ALIGNED_output); 
	free(XYZ_NOT_ALIGNED_input1);
	free(XYZ_NOT_ALIGNED_input2);
	free(XYZ_NOT_ALIGNED_output);

	free(XY_ALIGNED_input1);
	free(XY_ALIGNED_input2);
	free(XY_ALIGNED_output); 
	free(XY_NOT_ALIGNED_input1);
	free(XY_NOT_ALIGNED_input2);
	free(XY_NOT_ALIGNED_output);

	free(X_ALIGNED_input1);
	free(X_ALIGNED_input2);
	free(X_ALIGNED_output); 
	free(X_NOT_ALIGNED_input1);
	free(X_NOT_ALIGNED_input2);
	free(X_NOT_ALIGNED_output); 

	cudaFree(input1GPU);
	cudaFree(input2GPU);
	cudaFree(outputGPU);

	cudaFree(XYZ_ALIGNED_input1GPU);
	cudaFree(XYZ_ALIGNED_input2GPU);
	cudaFree(XYZ_ALIGNED_outputGPU);
	cudaFree(XYZ_NOT_ALIGNED_input1GPU);
	cudaFree(XYZ_NOT_ALIGNED_input2GPU);
	cudaFree(XYZ_NOT_ALIGNED_outputGPU);

	cudaFree(XY_ALIGNED_input1GPU);
	cudaFree(XY_ALIGNED_input2GPU);
	cudaFree(XY_ALIGNED_outputGPU);
	cudaFree(XY_NOT_ALIGNED_input1GPU);
	cudaFree(XY_NOT_ALIGNED_input2GPU);
	cudaFree(XY_NOT_ALIGNED_outputGPU);

	cudaFree(X_ALIGNED_input1GPU);
	cudaFree(X_ALIGNED_input2GPU);
	cudaFree(X_ALIGNED_outputGPU);
	cudaFree(X_NOT_ALIGNED_input1GPU);
	cudaFree(X_NOT_ALIGNED_input2GPU);
	cudaFree(X_NOT_ALIGNED_outputGPU);
}
