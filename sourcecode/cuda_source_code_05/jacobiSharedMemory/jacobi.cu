//=============================================================================================
// Name        		: jacobi.cu
// Author      		: Jose Refojo
// Version     		:	05-02-13
// Creation date	:	15-09-10
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will provide an estimate of a function integral in a given interval,
//			  the interval being provided by the user, but the function being fixed.
//=============================================================================================

#define BLOCK_SIZE 25
//#define MATRIX_SIZE 1000

int MATRIX_SIZE = 100;
int verbose = 0;
int skipCpuTest = 0;
int maxNumberOfIterations=100;


#include <getopt.h>
#include "stdio.h"
#include "time.h"


int		chooseCudaCard				();
void	cudaLastErrorCheck			(const char *message);
void	cudaTestBlockInformation	(dim3 myDimBlock);
void	cudaTestGridInformation		(dim3 myDimGrid);
int		parseArguments				(int argc, char *argv[]);
void	printUsage					(void);


// Careful here - this particular version is only valid if MATRIX_SIZE is divisible by BLOCK_SIZE, for simpler code!
__global__ void iterateGPUShared (int N,float *A1dGPU,float *bGPU,float *xOldGPU,float *xNewGPU) {	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int j,k;
	float sumatory;
	
	if (idx<N) {
		__shared__ float sharedxOldGPU[BLOCK_SIZE];
		sumatory=bGPU[idx];

		// We'll need to fetch the memory from xOldGPU into sharedxOldGPU at least N/BLOCK_SIZE times
		int chunk_size= BLOCK_SIZE;
		for (k=0;k<N;k+=chunk_size) {
			sharedxOldGPU[threadIdx.x]=xOldGPU[threadIdx.x+k];
        	__syncthreads();
			for (j=0;j<BLOCK_SIZE;j++) {
				if (idx!=j+k) {
					sumatory-=(A1dGPU[j+k+idx*N]*sharedxOldGPU[j]);
				}
			}
		}
		sumatory*=(1.0f/A1dGPU[idx+idx*N]);
		xNewGPU[idx]=sumatory;
	}
}



__global__ void iterateGPU (int N,float *A1dGPU,float *bGPU,float *xOldGPU,float *xNewGPU) {	
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int j;
	float sumatory;

	if (idx<N) {
		sumatory=bGPU[idx];
		for (j=0;j<N;j++) {
		   if (idx!=j) {
			sumatory-=(A1dGPU[j+idx*N]*xOldGPU[j]);
		   }
		}
		sumatory*=(1.0f/A1dGPU[idx+idx*N]);
		xNewGPU[idx]=sumatory;
	}
}
void iterateCPU (int N,float **A,float *b,float *xOld,float *xNew) {
	int i,j;
	float sumatory;
	for (i=0;i<N;i++) {
		sumatory=b[i];
		for (j=0;j<N;j++) {
		   if (i!=j)
			sumatory-=(A[i][j]*xOld[j]);
		}
		sumatory*=(1.0f/A[i][i]);
		xNew[i]=sumatory;
	}

	if (verbose) {
		printf("Before iteration %d, xOld=(",i);
		for (j=0;j<N;j++) {
			printf("%f",xOld[j]);
			if (j+1<N) {
				printf(",");
			}
		}
		printf(")\n");
		printf("After iteration %d, xNew=(",i);
		for (j=0;j<N;j++) {
			printf("%f",xNew[j]);
			if (j+1<N) {
				printf(",");
			}
		}
		printf(")\n");
	}
}
bool checkSolution (int N,float **A,float *b,float *xNew) {
	// Calculate r=Ax-b and see how far from [0,,0] it is
	float *r;
	r = (float*) malloc( N*sizeof(float) );
	int i,j;
	float tmpNorm=0.0f;

	for (i=0;i<N;i++) {
		r[i]=-b[i];
		for (j=0;j<N;j++) {
			r[i]+=A[i][j]*xNew[j];
		}
		tmpNorm += r[i]*r[i];
	}
	free(r);

	printf("checkSolution, tmpNorm: %f\n",tmpNorm);
	if (tmpNorm<1.E-5) {
		return true;
	} else {
		return false;
	}
}
int main (int argc, char *argv[]) {
	int i,j;
	// Serial Test first:
	float CPUTime,GPUTime;

	parseArguments(argc, argv);
	chooseCudaCard();

	int N = MATRIX_SIZE;

	// Matrix A
	float *A1d;
	float *A1dGPU;
	float **A;
	A1d = (float*) malloc( N*N*sizeof(float) );
	A = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		A[i]=(&(A1d[i*N]));
	}
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			if (i!=j) {
				A[i][j] = 0.1;
			} else {
				A[i][j] = 20*N;
			}
		}
	}

	cudaMalloc ((void **) &A1dGPU, sizeof(float)*(N*N));
	cudaLastErrorCheck("(Cuda error cudaMalloc A1dGPU)");
	cudaMemcpy(A1dGPU, A1d, sizeof(float)*(N*N), cudaMemcpyHostToDevice);
	cudaLastErrorCheck("(Cuda error cudaMemcpy A1dGPU)");

	// Vectors b,xOld,xNew
	float *b,*xOld,*xNew;
	float *bGPU,*xOldGPU,*xNewGPU;
	b = (float*) malloc( N*sizeof(float) );
	for (i=0;i<N;i++) {
		b[i]=i;
	}
	xOld = (float*) malloc( N*sizeof(float) );
	xNew = (float*) malloc( N*sizeof(float) );
	cudaMalloc ((void **) &bGPU, sizeof(float)*N);
	cudaMalloc ((void **) &xOldGPU, sizeof(float)*N);
	cudaMalloc ((void **) &xNewGPU, sizeof(float)*N);

	// We set up the first step of the method as (1,2,...,N)
	for (int i=0;i<N;i++)
		xOld[i]=i+1;
	
	cudaMemcpy(bGPU   ,    b, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xNewGPU, xNew, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xOldGPU, xOld, sizeof(float)*N, cudaMemcpyHostToDevice);

	printf("***********************************************************************************************\n");
	printf("******** This program will provide an estimate of the solution of a linear system problem, ****\n");
	printf("******** using the Jacobi method                                                           ****\n");
	printf("***********************************************************************************************\n");

	printf("======>Problem size: %d\n",MATRIX_SIZE);

	clock_t jacobiCPUStart = clock();

	// Iterate
	if (!skipCpuTest) {
		for (i=0;i<maxNumberOfIterations;i++) {
			printf("======>Iteration %d in CPU\n",i);
			iterateCPU(N,A,b,xOld,xNew);
			if ( checkSolution (N,A,b,xNew) ) {
				// Convergence
				printf("Convergence in %d iterations with the SERIAL code\n",i);
				if (verbose) {
					printf("The solution found was:\n");
					for (j=0;j<N;j++) {
						printf("xNew[%d]=%f\n",j,xNew[j]);
					}
				}
				break;
			} else {
				// No convergence yet, we move xNew to xOld and start again
				for (int i=0;i<N;i++)
					xOld[i]=xNew[i];
				printf("No convergence in the CPU\n");
			}
		}
	} else {
	printf("CPU test skipped\n");
	}

	CPUTime=(float)(clock()-jacobiCPUStart)/(float)(CLOCKS_PER_SEC);
	printf("CPU test took: %f seconds\n",CPUTime);

	clock_t jacobiGPUStart = clock();

	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	// Test block and grid
	cudaTestBlockInformation (dimBlock);
	cudaLastErrorCheck("(Cuda error cudaTestBlockInformation)");
	cudaTestGridInformation (dimGrid);
	cudaLastErrorCheck("(Cuda error cudaTestGridInformation)");

	for (i=0;i<maxNumberOfIterations;i++) {



//		iterateGPU<<<dimGrid,dimBlock>>>(N,A1dGPU,bGPU,xOldGPU,xNewGPU);
		iterateGPUShared<<<dimGrid,dimBlock>>>(N,A1dGPU,bGPU,xOldGPU,xNewGPU);
		cudaLastErrorCheck("(Cuda error in iterateGPU)");

		cudaMemcpy(xNew, xNewGPU, sizeof(float)*N, cudaMemcpyDeviceToHost);

		printf("======>Iteration %d in GPU\n",i);
		if ( checkSolution (N,A,b,xNew) ) {
			// Convergence
			printf("Convergence in %d iterations with the CUDA code\n",i);
			if (verbose) {
				printf("The solution found was:\n");
				for (j=0;j<N;j++) {
					printf("xNew[%d]=%f\n",j,xNew[j]);
				}
			}
			break;
		} else {
			// No convergence yet, we move xNew to xOld and start again
			for (int i=0;i<N;i++)
				xOld[i]=xNew[i];
			cudaMemcpy(xOldGPU, xOld, sizeof(float)*N, cudaMemcpyHostToDevice);
			printf("No convergence GPU\n");
		}
	}

	GPUTime=(float)(clock()-jacobiGPUStart)/(float)(CLOCKS_PER_SEC);
	printf("GPU test took: %f seconds\n",GPUTime);
	printf("\n");


	free(A);
	free(A1d);
	cudaFree(A1dGPU);

	free(b);
	free(xOld);
	free(xNew);

	cudaFree(bGPU);
	cudaFree(xOldGPU);
	cudaFree(xNewGPU);
}


// Choose card to use - will find all the cards and choose the one with more multi-processors
int chooseCudaCard() {
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
			case 1:
				numberOfCUDAcoresForThisCC=8;
				break;
			case 2:
				numberOfCUDAcoresForThisCC=48;
				break;
			case 3:
				numberOfCUDAcoresForThisCC=192;
				break;
			default:
				numberOfCUDAcoresForThisCC=0;	//???
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
		printf("GPU maximum 2D texture size: %d %d\n",x.maxTexture2D[0],x.maxTexture2D[1]);
		printf("GPU maximum 3D texture size: %d %d %d\n",x.maxTexture3D[0],x.maxTexture3D[1],x.maxTexture3D[2]);
		printf("GPU maximum 1D layered texture dimensions: %d %d\n",x.maxTexture1DLayered[0],x.maxTexture1DLayered[1]);
		printf("GPU maximum 2D layered texture dimensions: %d %d %d\n",x.maxTexture2DLayered[0],x.maxTexture2DLayered[1],x.maxTexture2DLayered[2]);
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
		}
	} else { // 2d case
		if (myDimBlock.x*myDimBlock.y<=x.maxThreadsPerBlock) {
			printf("The GPU can support this block size\n");
		} else {
			printf("The GPU can NOT support this block size\n");
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
		}
	} else { // 2d case
		if ( (myDimGrid.x<=x.maxGridSize[0])&&(myDimGrid.y<=x.maxGridSize[1])) {
			printf("The GPU can support this grid size\n");
		} else {
			printf("The GPU can NOT support this grid size\n");
		}
	}
}


int parseArguments (int argc, char *argv[]) {
	int c;
	int tmpInt;

	while ((c = getopt (argc, argv, "hi:n:sv")) != -1) {
		switch(c) {
			case 'h':
				printUsage(); break;
			case 'i':
				tmpInt = atoi(optarg);
				if (tmpInt>0) {
					maxNumberOfIterations = tmpInt;
				}
				break;
			case 'n':
				tmpInt = atoi(optarg);
				if (tmpInt>0) {
					MATRIX_SIZE = tmpInt;
				}
				break;
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
	printf(" Name                 : jacobi.cu\n");
	printf(" Author               : Jose Mauricio Refojo <jose@tchpc.tcd.ie>\n");
	printf(" Version              : 1.0d\n");
	printf(" Creation date        :	15-09-10\n");
	printf(" Copyright            : Copyright belongs to Trinity Centre for High Performance Computing\n");
	printf(" Description          : This program will generate and solve a problem using a jacobi numberical solver\n");
	printf("                        in both the CPU and the GPU\n");
	printf("usage:\n");
	printf("jacobi [options]\n");
	printf("      -h           : will show this usage\n");
	printf("      -i   number  : will set the maximum number of iterations to number (default: %d)\n",maxNumberOfIterations);
	printf("      -n   size    : will set the number of rows of the first column to size (default: %d)\n",MATRIX_SIZE);
	printf("      -s           : will skip the CPU test\n");
	printf("      -v           : will run in verbose mode\n");
	printf("=============================================================================================");
	printf("     \n");
	exit(0);
}
