//=============================================================================================
// Name        		: matrixMultiplication.cu
// Author      		: Jose Refojo
// Version     		:	09-02-2024
// Creation date	:	22-09-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will multiply two matrices into a third one
//=============================================================================================

// Remember that we are using a 2d grid, so the total number of threads per block will be BLOCK_SIZExBLOCK_SIZE
#define BLOCK_SIZE 8
//#define BLOCK_SIZE 2048
int MATRIX_SIZE_N = 2000;
int MATRIX_SIZE_M = 2000;
int verbose = 0;
int skipCpuTest = 0;

#include "stdio.h"
#include "time.h"
#include <getopt.h>

int	chooseCudaCard			(bool verbose);
void	cudaLastErrorCheck		(const char *message);
void	cudaTestBlockInformation	(dim3 myDimBlock);
void	cudaTestGridInformation		(dim3 myDimGrid);
int	parseArguments			(int argc, char **argv);
void	printUsage			(void);

__global__ void computeMatMulGPU (int N,int M,float *A1dGPU,float *B1dGPU,float *C1dGPU) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int k;

	if (idx<N) {
		if (idy<N) {
			C1dGPU[idx+idy*N]=0.0f;
			for (k=0;k<M;k++) {
				C1dGPU[idx+idy*N]+=A1dGPU[k+idy*M]*B1dGPU[k+idx*M];
			}
		}
	}
}
void matmulGPU (int N,int M,float **A,float *A1dGPU,float **B,float *B1dGPU,float **C,float *C1dGPU) {
	int i,j;
	cudaError_t err;

	// Matrix BTrans, the transpose of B
	float *BTrans1d;
	float *BTrans1dGPU;
	float **BTrans;
	BTrans1d = (float*) malloc( N*M*sizeof(float) );
	BTrans = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		BTrans[i]=(&(BTrans1d[i*M]));
	}
	for (i=0;i<N;i++) {
		for (j=0;j<M;j++) {
			BTrans[i][j]=B[j][i];
		}
	}

	printf("Amount of bytes needed for BTrans1dGPU %lu (%lu in KBs, %lu in MBs)\n",N*M*sizeof(float),(N*M*sizeof(float))/1024,(N*M*sizeof(float))/1048576);


	err = cudaMalloc ((void **) &BTrans1dGPU, sizeof(float)*(N*M));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMalloc BTrans1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(BTrans1dGPU, BTrans1d, sizeof(float)*(N*M), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMemcpy BTrans1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	int block_size=BLOCK_SIZE;

	dim3 dimBlock(block_size,block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(N/dimBlock.y) + (!(N%dimBlock.y)?0:1) );

	// Test block and grid
	cudaTestBlockInformation (dimBlock);
	cudaLastErrorCheck("(Cuda error cudaTestBlockInformation)");
	cudaTestGridInformation (dimGrid);
	cudaLastErrorCheck("(Cuda error cudaTestGridInformation)");

	printf("Block size test (2d): %dx%d\n",dimBlock.x,dimBlock.y);
	printf("Grid size in each dimension: %dx%dx%d\n",dimGrid.x,dimGrid.y,dimGrid.z);

	computeMatMulGPU<<<dimGrid,dimBlock>>>(N,M,A1dGPU,BTrans1dGPU,C1dGPU);
	cudaLastErrorCheck("(Cuda error in computeMatMulGPU)");

	err = cudaMemcpy(&(C[0][0]), C1dGPU, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMemcpy C1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
/*
	// CPU computation, just in case
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j]=0.0f;
			for (k=0;k<M;k++) {				
				C[i][j]+=A[i][k]*BTrans[j][k];
			}
		}
	}
*/
	free(BTrans);
	free(BTrans1d);
	err = cudaFree(BTrans1dGPU);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMemcpy BTrans1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
}
void matmulCPU (int N,int M,float **A,float **B,float **C) {
	int i,j,k;
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j]=0.0f;
			for (k=0;k<M;k++) {				
				C[i][j]+=A[i][k]*B[k][j];
			}
		}
	}
}
int main(int argc, char *argv[]) {
	int i,j;
	cudaError_t err;

	parseArguments(argc, argv);
	chooseCudaCard(verbose);


	// Serial Test first:
	int N = MATRIX_SIZE_N;
	int M = MATRIX_SIZE_M;

	// Matrix A
	float *A1d;
	float *A1dGPU;
	float **A;
	A1d = (float*) malloc( N*M*sizeof(float) );
	A = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		A[i]=(&(A1d[i*M]));
	}

	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<M;j++) {
			A[i][j] = i+j+1.0;
		}
	}
/*
	A[0][0] = 1;	A[0][1] = 1;	A[0][2] = 1;
	A[1][0] = 1;	A[1][1] = 1;	A[1][2] = 1;
*/
	err = cudaMalloc ((void **) &A1dGPU, sizeof(float)*(N*M));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc A1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(A1dGPU, A1d, sizeof(float)*(N*M), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMemcpy A1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	// Matrix B
	float *B1d;
	float *B1dGPU;
	float **B;
	B1d = (float*) malloc( M*N*sizeof(float) );
	B = (float**) malloc(M*sizeof(float*));
	for (i=0;i<M;i++) {
		B[i]=(&(B1d[i*N]));
	}
	// Initialization
	for (i=0;i<M;i++) {
		for (j=0;j<N;j++) {
			B[i][j] = (j+1);
		}
	}
/*
	B[0][0] = 1;	B[0][1] = 1;
	B[1][0] = 1;	B[1][1] = 1;
	B[2][0] = 1;	B[2][1] = 1;

	B[0][0] = 2;	B[0][1] = 2;
	B[1][0] = 2;	B[1][1] = 2;
	B[2][0] = 2;	B[2][1] = 2;
*/
	err = cudaMalloc ((void **) &B1dGPU, sizeof(float)*(M*N));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc B1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(B1dGPU, B1d, sizeof(float)*(M*N), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMemcpy B1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	// Matrix C
	float *C1d;
	float *C1dGPU;
	float **C;
	C1d = (float*) malloc( N*N*sizeof(float) );
	C = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		C[i]=(&(C1d[i*N]));
	}
	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j] = 0;
		}
	}
/*
	C[0][0] = -1;	C[0][1] = -1;	C[0][2] = -1;
	C[1][0] = -1;	C[1][1] = -1;	C[1][2] = -1;
*/
	err = cudaMalloc ((void **) &C1dGPU, sizeof(float)*(N*N));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc C1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}


	float CPUTime,GPUTime;

	printf("***********************************************************************************************\n");
	printf("******** This program will calculate the multiplication of two hard coded matrices         ****\n");
	printf("******** of sizes: A= %d x %d ; b= %d x %d; block size= %d        ****\n",N,M,M,N,BLOCK_SIZE);
	printf("***********************************************************************************************\n");

	clock_t matmulCPUStart = clock();

	if (!skipCpuTest) matmulCPU(N,M,A,B,C);

	CPUTime=(float)(clock()-matmulCPUStart)/(float)(CLOCKS_PER_SEC);

	if (verbose) {
		printf("Matrix A:\n");
		for (i=0;i<N;i++) {
			for (j=0;j<M;j++) {
				printf("%f ",A[i][j]);
			}
			printf("\n");
		}
		printf("\n");

		printf("Matrix B:\n");
		for (i=0;i<M;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",B[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("******************************    matmulCPU    ***************************************\n");
	printf("Matrix C (calculated in the CPU):\n"); 
	if (verbose) {
		for (i=0;i<N;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",C[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	} else {
		printf(" C[%d][%d]=%f ",0,0,C[0][0]);
		printf(" C[%d][%d]=%f ",N-1,0,C[N-1][0]);
		printf(" C[%d][%d]=%f ",0,N-1,C[0][N-1]);
		printf(" C[%d][%d]=%f\n",N-1,N-1,C[N-1][N-1]);
	}

	printf("matrix multiplication on the CPU took: %f seconds\n",CPUTime);

	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j] = 0;
		}
	}

	clock_t matmulGPUStart = clock();

	matmulGPU (N,M,A,A1dGPU,B,B1dGPU,C,C1dGPU);

	GPUTime=(float)(clock()-matmulGPUStart)/(float)(CLOCKS_PER_SEC);

	printf("******************************    matmulGPU    ***************************************\n");
	printf("Matrix C (calculated in the GPU):\n"); 
	if (verbose) {
		for (i=0;i<N;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",C[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	} else {
		printf(" C[%d][%d]=%f ",0,0,C[0][0]);
		printf(" C[%d][%d]=%f ",N-1,0,C[N-1][0]);
		printf(" C[%d][%d]=%f ",0,N-1,C[0][N-1]);
		printf(" C[%d][%d]=%f\n",N-1,N-1,C[N-1][N-1]);
	}

	printf("matrix multiplication on the GPU took: %f seconds\n",GPUTime);
	
	printf("******************************    speedup: %f    ***************************************\n",(CPUTime/GPUTime));

	free(A);
	free(A1d);
	cudaFree(A1dGPU);

	free(B);
	free(B1d);
	cudaFree(B1dGPU);

	free(C);
	free(C1d);
	cudaFree(C1dGPU);
}



// Choose card to use - will find all the cards and choose the one with more multi-processors
int chooseCudaCard(bool verbose) {
	int i,numberOfDevices,best,bestNumberOfMultiprocessors;
	int numberOfCUDAcoresForThisCC=0;
	struct cudaDeviceProp x;

	if ( cudaGetDeviceCount(&numberOfDevices)!=cudaSuccess ) {
		printf("No CUDA-enabled devices were found\n");
	}
	printf("***************************************************\n");
	printf("Found %d CUDA-enabled devices\n",numberOfDevices);
	best=-1;
	bestNumberOfMultiprocessors=-1;
	for (i=0;i<numberOfDevices;i++) {
		cudaGetDeviceProperties(&x, i);
		printf("========================= IDENTITY DATA ==================================\n");
		printf("GPU model name: %s\n",x.name);
		if (x.integrated==1) {
			printf("GPU The device is an integrated (motherboard) GPU\n");
		} else {
			printf("GPU The device is NOT an integrated (motherboard) GPU - i.e. it is a discrete device\n");
		}
		printf("GPU pciBusID: %d\n",x.pciBusID);
		printf("GPU pciDeviceID: %d\n",x.pciDeviceID);
		printf("GPU pciDomainID: %d\n",x.pciDomainID);
		if (x.tccDriver==1) {
			printf("the device is a Tesla one using TCC driver\n");
		} else {
			printf("the device is NOT a Tesla one using TCC driver\n");
		}
		printf("========================= COMPUTE DATA ==================================\n");
		printf("GPU Compute capability: %d.%d\n",x.major,x.minor);
		switch (x.major) {
			case 1:	// Tesla / T10
				numberOfCUDAcoresForThisCC=8;
				break;
			case 2:	// Fermi
				numberOfCUDAcoresForThisCC=32;
				break;
			case 3:	// Kepler
				numberOfCUDAcoresForThisCC=192;
				break;
			case 5:	// Maxwell
				numberOfCUDAcoresForThisCC=128;
				break;
			case 6:	// Pascal
				switch (x.minor) {
					case 0: // GP100, 64 cuda cores per SM - 7.0 should be prefered over 7.1
						numberOfCUDAcoresForThisCC=64;
						break;
					case 1: // GP102, GP104, GP106, GP107, 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					default: // Unknown
						numberOfCUDAcoresForThisCC=0;
						break;
				}
				numberOfCUDAcoresForThisCC=128;
				break;
			case 7:	// Volta is 7.0 and 7.2, 64 cuda cores per SM, Turing is 7.5 - also has 64 cuda cores per SM
				numberOfCUDAcoresForThisCC=64;
				break;
			case 8:	// Ampere 8.x, with x < 9, has 64 cuda cores per SM, but Ada Lovelace (8.9) has 128 cuda cores per SM
				switch (x.minor) {
					case 0: // GP100, 64 cuda cores per SM - 7.0 should be prefered over 7.1
						numberOfCUDAcoresForThisCC=64;
						break;
					case 9: // GP10B, Pascal Tegra cards  - still 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
						numberOfCUDAcoresForThisCC=64;
						break;
				}
				break;
			case 9:	// Hopper (G100 is 9.0) and Grace Hopper both have 128 cuda cores per SM
				numberOfCUDAcoresForThisCC=128;
				break;
            case 10: // Blackwell has 128 cuda cores per SM
    			numberOfCUDAcoresForThisCC=128;
				break;
			default: // Unknown
				numberOfCUDAcoresForThisCC=0;
				break;
		}
		if (x.multiProcessorCount>bestNumberOfMultiprocessors*numberOfCUDAcoresForThisCC) {
			best=i;
			bestNumberOfMultiprocessors=x.multiProcessorCount*numberOfCUDAcoresForThisCC;
		}
		printf("GPU Clock frequency in hertzs: %d\n",x.clockRate);
		printf("GPU Device can concurrently copy memory and execute a kernel: %d\n",x.deviceOverlap);
		printf("GPU number of multi-processors: %d\n",x.multiProcessorCount);
		printf("GPU maximum number of threads per multi-processor: %d\n",x.maxThreadsPerMultiProcessor);
		printf("GPU Maximum size of each dimension of a grid: %dx%dx%d\n",x.maxGridSize[0],x.maxGridSize[1],x.maxGridSize[2]);
		printf("GPU Maximum size of each dimension of a block: %dx%dx%d\n",x.maxThreadsDim[0],x.maxThreadsDim[1],x.maxThreadsDim[2]);
		printf("GPU Maximum number of threads per block: %d\n",x.maxThreadsPerBlock);
		printf("GPU Maximum pitch in bytes allowed by memory copies: %lu\n",x.memPitch);
		printf("GPU Compute mode is: %d\n",x.computeMode);
		printf("========================= MEMORY DATA ==================================\n");
		printf("GPU total global memory: %lu bytes\n",x.totalGlobalMem);
		printf("GPU peak memory clock frequency in kilohertz: %d bytes\n",x.memoryClockRate);
		printf("GPU memory bus width: %d bits\n",x.memoryBusWidth);
		printf("GPU L2 cache size: %d bytes\n",x.l2CacheSize);
		printf("GPU 32-bit registers available per block: %d\n",x.regsPerBlock);
		printf("GPU Shared memory available per block in bytes: %lu\n",x.sharedMemPerBlock);
		printf("GPU Alignment requirement for textures: %lu\n",x.textureAlignment);
		printf("GPU Constant memory available on device in bytes: %lu\n",x.totalConstMem);
		printf("GPU Warp size in threads: %d\n",x.warpSize);
		printf("GPU maximum 1D texture size: %d\n",x.maxTexture1D);
		printf("GPU maximum 2D texture size: %d\n",x.maxTexture2D[0],x.maxTexture2D[1]);
		printf("GPU maximum 3D texture size: %d\n",x.maxTexture3D[0],x.maxTexture3D[1],x.maxTexture3D[2]);
		printf("GPU maximum 1D layered texture dimensions: %d\n",x.maxTexture1DLayered[0],x.maxTexture1DLayered[1]);
		printf("GPU maximum 2D layered texture dimensions: %d\n",x.maxTexture2DLayered[0],x.maxTexture2DLayered[1],x.maxTexture2DLayered[2]);
		printf("GPU surface alignment: %lu\n",x.surfaceAlignment);
		if (x.canMapHostMemory==1) {
			printf("GPU The device can map host memory into the CUDA address space\n");
		} else {
			printf("GPU The device can NOT map host memory into the CUDA address space\n");
		}
		if (x.ECCEnabled==1) {
			printf("GPU memory has ECC support\n");
		} else {
			printf("GPU memory does not have ECC support\n");
		}
		if (x.ECCEnabled==1) {
			printf("GPU The device shares an unified address space with the host\n");
		} else {

			printf("GPU The device DOES NOT share an unified address space with the host\n");
		}
		printf("========================= EXECUTION DATA ==================================\n");
		if (x.concurrentKernels==1) {
			printf("GPU Concurrent kernels are allowed\n");
		} else {
			printf("GPU Concurrent kernels are NOT allowed\n");
		}
		if (x.kernelExecTimeoutEnabled==1) {
			printf("GPU There is a run time limit for kernels executed in the device\n");
		} else {
			printf("GPU There is NOT a run time limit for kernels executed in the device\n");
		}
		if (x.asyncEngineCount==1) {
			printf("GPU The device can concurrently copy memory between host and device while executing a kernel\n");
		} else if (x.asyncEngineCount==2) {
			printf("GPU The device can concurrently copy memory between host and device in both directions and execute a kernel at the same time\n");
		} else {
			printf("GPU the device is NOT capable of concurrently memory copying\n");
		}
	}
	// set the best device
	if (best>=0) {
		cudaGetDeviceProperties(&x, best);
		printf("Choosing %s\n", x.name);
		cudaSetDevice(best);
	}
	// We return the number of devices, in case we want to use more than one
	printf("***************************************************\n");
	return (numberOfDevices);
}

void cudaLastErrorCheck (const char *message) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n",message,cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
}

void cudaTestBlockInformation (dim3 myDimBlock) {
	int currentDevice;
	struct cudaDeviceProp x;

	cudaGetDevice(&currentDevice) ;
	cudaGetDeviceProperties(&x, currentDevice);
	printf("GPU Maximum size of each dimension of a block: %dx%dx%d\n",x.maxThreadsDim[0],x.maxThreadsDim[1],x.maxThreadsDim[2]);
   	printf("Current grid size: %dx%dx%d\n",myDimBlock.x,myDimBlock.y,myDimBlock.z);
	if (myDimBlock.y==1) {	// 1d case
		if (myDimBlock.x<=x.maxThreadsPerBlock) {
			printf("The GPU can support this block size\n");
		} else {
			printf("The GPU can NOT support this block size\n");
			exit(1);
		}
	} else { // 2d case
		if (myDimBlock.x*myDimBlock.y<=x.maxThreadsPerBlock) {
			printf("The GPU can support this block size\n");
		} else {
			printf("The GPU can NOT support this block size\n");
			exit(1);
		}

	}
}

void cudaTestGridInformation (dim3 myDimGrid) {
	int currentDevice;
	struct cudaDeviceProp x;

	cudaGetDevice(&currentDevice) ;
	cudaGetDeviceProperties(&x, currentDevice);
   	printf("GPU Maximum size of each dimension of a grid: %dx%dx%d\n",x.maxGridSize[0],x.maxGridSize[1],x.maxGridSize[2]);
   	printf("Current grid size: %dx%dx%d\n",myDimGrid.x,myDimGrid.y,myDimGrid.z);
	if (myDimGrid.y==1) {	// 1d case
		if (myDimGrid.x<=x.maxGridSize[0]) {
			printf("The GPU can support this grid size\n");
		} else {
			printf("The GPU can NOT support this grid size\n");
			exit(1);
		}
	} else { // 2d case
		if ( (myDimGrid.x<=x.maxGridSize[0])&&(myDimGrid.y<=x.maxGridSize[1])) {
			printf("The GPU can support this grid size\n");
		} else {
			printf("The GPU can NOT support this grid size\n");
			exit(1);
		}
	}
}


int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "hn:m:sv")) != -1) {
		switch(c) {
			case 'h':
				printUsage(); break;
			case 'n':
				 MATRIX_SIZE_N = atoi(optarg); break;
			case 'm':
				 MATRIX_SIZE_M = atoi(optarg); break;
			case 's':
				skipCpuTest = 1; break;
			case 'v':
				verbose = 1; break;
			default:
				fprintf(stderr, "Invalid option given\n");
				return -1;
		}	
	}
	return 0;
}
void printUsage () {
	printf("=============================================================================================\n");
	printf(" Name                 : matrixMultiplication.cu\n");
	printf(" Author               : Jose Mauricio Refojo <jose@tchpc.tcd.ie>\n");
	printf(" Version              : 1.0d\n");
	printf(" Creation date        :	01-02-13\n");
	printf(" Copyright            : Copyright belongs to Trinity Centre for High Performance Computing\n");
	printf(" Description          : This program will calculate the product matrix of two matrices of sizes\n");
	printf("                        nxm and mxn, by using the global memory in the GPU\n");
	printf("usage:\n");
	printf("matrixMultiplicationTexture [options]\n");
	printf("      -h           : will show this usage\n");
	printf("      -n   size    : will set the number of rows of the first column to size (default: 100)\n");
	printf("      -m   size    : will set the number of columns of the first column to size (default: 100)\n");
	printf("      -s           : will skip the CPU test\n");
	printf("      -v           : will run in verbose mode\n");
	printf("=============================================================================================");
	printf("     \n");
}
