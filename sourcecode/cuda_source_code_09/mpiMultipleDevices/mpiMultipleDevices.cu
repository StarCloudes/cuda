//=============================================================================================
// Name        		: mpiMultipleDevices.cu
// Author      		: Jose Refojo
// Version     		:	24-03-24
// Creation date	:	25-03-24
//=============================================================================================

#define MATRIX_SIZE_N 40
#define MATRIX_SIZE_M 20

#include "stdio.h"
#include "time.h"
#include "mpi.h"

#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>

#define BLOCK_SIZE 16
#define PROBLEM_SIZE 12

int findBestDevice ();

bool verbose = false;

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int numberOfElements) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < numberOfElements ) {
		out[idx]=in1[idx]+in2[idx];
		//out[idx]=idx;
	}
}
int main(int argc, char *argv[]) {
	int         my_rank;			// rank of process
	int         number_of_processes;	// number of processes

	// Start up MPI
	MPI_Init(&argc, &argv);

	// Find out process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Find out number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

	int N=PROBLEM_SIZE;
	int i;
	int from,to;
	int numberOfElementsPerNode = N/number_of_processes;
	int remainder= N-number_of_processes*numberOfElementsPerNode;

	int *numberOfElements=NULL;	// Will store the index with each one of the numberOfElements
	numberOfElements = (int*) malloc(number_of_processes*sizeof(int));
	if (numberOfElements==NULL) {
		printf ("Ran out of memory when trying to allocate 'numberOfElements'\n");
		exit (1);
	}

	int *displacements=NULL;
	displacements = (int*) malloc(number_of_processes*sizeof(int));
	if (displacements==NULL) {
		printf ("Ran out of memory when trying to allocate 'displacements'\n");
		exit (1);
	}

	// Find out from where to where this process is computing
	from=my_rank*numberOfElementsPerNode;
	to=(my_rank+1)*numberOfElementsPerNode;	
	if (my_rank==(number_of_processes-1)) {
		to+=remainder;
	}
	int numberOfElementsInThisNode = (to-from);

	// All the nodes need to know where the others are computing so we can transfer data correctly
	for (i=0; i<number_of_processes; i++) {
		numberOfElements[i]=numberOfElementsPerNode;
	}
	numberOfElements[number_of_processes-1]+=remainder;
	displacements[0]=0;
	for (i=1; i<number_of_processes; i++) {
		displacements[i]=displacements[i-1]+numberOfElements[i-1];
	}
	if (my_rank==0) {
		printf("This is process %d, from=%d to=%d remainder=%d numberOfElementsInThisNode=%d\n", my_rank,from,to,remainder,numberOfElementsInThisNode);
		for (i=0; i<number_of_processes; i++) {
			printf("numberOfElements[%d]=%d\n",i,numberOfElements[i]);
		}
		for (i=0; i<number_of_processes; i++) {
			printf("displacements[%d]=%d\n",i,displacements[i]);
		}

	}

	// Set up and configute the card 
	int cardForThisProcess;
	int numberOfDevices = findBestDevice();
	if (numberOfDevices>1) {
		cardForThisProcess=my_rank%numberOfDevices;
		printf("This is process %d, numberOfDevices = %d cardForThisProcess=%d\n", my_rank,numberOfDevices,cardForThisProcess);		
		cudaSetDevice(cardForThisProcess);
	}

	// pointers to host memory
	float *a, *b, *c;
	// pointers to device memory
	float *a_d, *b_d, *c_d;

	// Allocate arrays a, b and c on host
	a = (float*) malloc(N*sizeof(float));
	b = (float*) malloc(N*sizeof(float));
	c = (float*) malloc(N*sizeof(float));

	// Allocate arrays a_d, b_d and c_d on device
	cudaMalloc ((void **) &a_d, sizeof(float)*numberOfElementsInThisNode);
	cudaMalloc ((void **) &b_d, sizeof(float)*numberOfElementsInThisNode);
	cudaMalloc ((void **) &c_d, sizeof(float)*numberOfElementsInThisNode);

	// Initialize arrays a and b
	for (i=0; i<N; i++) {
		a[i]= (float) (i);
		b[i]= (float) (2*i);
		c[i]= 0.0;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(a_d, &(a[from]), sizeof(float)*numberOfElementsInThisNode, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, &(b[from]), sizeof(float)*numberOfElementsInThisNode, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	add_arrays_gpu<<<dimGrid,dimBlock>>>(a_d, b_d, c_d, numberOfElementsInThisNode);


	// Copy data from device memory to host memory
	cudaMemcpy(&(c[from]), c_d, sizeof(float)*numberOfElementsInThisNode, cudaMemcpyDeviceToHost);
/*
	for (i=from; i<to; i++) {
		c[i]=i+1000*my_rank;
	}
*/

	if (my_rank==0) {
		printf("This is process %d, c (pre 2)=\n", my_rank);
		for (i=0; i<N; i++) {
			printf("c[%3d]=%10f\n",i,c[i]);
		}

	}

	// All processes need to know all other values - we sort it out with MPI_Allgatherv
	MPI_Allgatherv(
		&(c[from]),		// void* sendbuf,			// in
		(to-from),		// int sendcount,			// in
		MPI_FLOAT,		// MPI_Datatype sendtype,		// in
		&(c[0]),		// void* recvbuf,			// out
		&(numberOfElements[0]),		  						// int* recvcounts,			// in
		&(displacements[0]),    								// int* displs,				// in
		MPI_FLOAT,		// MPI_Datatype recvtype,	// in
		MPI_COMM_WORLD);	// MPI_Comm comm);			// in


	if (my_rank==0) {
		printf("This is process %d, c (post)=\n", my_rank);
		for (i=0; i<N; i++) {
			printf("c[%3d]=%10f\n",i,c[i]);
		}

	}

	// Free the memory
	if (a!=NULL) free(a);
	if (b!=NULL) free(b);
	if (c!=NULL) free(c);
	if (numberOfElements!=NULL) free(numberOfElements);
	if (displacements!=NULL) free(displacements);
	cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);

	// Shut down MPI
	MPI_Finalize();
}

int findBestDevice() {
	int i,numberOfDevices,best,bestNumberOfMultiprocessors;
	int numberOfCUDAcoresForThisCC=0;
	struct cudaDeviceProp x;

	if ( cudaGetDeviceCount(&numberOfDevices)!=cudaSuccess ) {
		printf("No CUDA-enabled devices were found\n");
	}
	if (verbose) {
		printf("***************************************************\n");
		printf("Found %d CUDA-enabled devices\n",numberOfDevices);
	}
	best=-1;
	bestNumberOfMultiprocessors=-1;
	for (i=0;i<numberOfDevices;i++) {
		cudaGetDeviceProperties(&x, i);
		if (verbose) {
			printf("Device %d - GPU model name: %s\n",i,x.name);
		}
		if (verbose) {
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
					numberOfCUDAcoresForThisCC=32;
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
	}
	// set the best device
	if (best>=0) {
		cudaGetDeviceProperties(&x, best);
		if (verbose) {
			printf("Choosing %s\n", x.name);
		}
		cudaSetDevice(best);
	}
	// We return the number of devices, in case we want to use more than one
	if (verbose) {
		printf("***************************************************\n");
	}
	return (numberOfDevices);
}
