//=============================================================================================
// Name        		: cudaUtils.cu
// Author      		: Jose Refojo
// Version     		:	09-02-2024
// Creation date	:	06-02-2018
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This source code contains common utils for cuda programs
//=============================================================================================

#include "stdio.h"
#include "cudaUtils.h"

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
		}

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

		if (verbose) {
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
	if (myDimBlock.x*myDimBlock.y*myDimBlock.z<=x.maxThreadsPerBlock) {
		printf("The GPU can support this block size\n");
	} else {
		printf("The GPU can NOT support this block size\n");
		exit(1);
	}
}

void cudaTestGridInformation (dim3 myDimGrid) {
	int currentDevice;
	struct cudaDeviceProp x;

	cudaGetDevice(&currentDevice) ;
	cudaGetDeviceProperties(&x, currentDevice);
   	printf("GPU Maximum size of each dimension of a grid: %dx%dx%d\n",x.maxGridSize[0],x.maxGridSize[1],x.maxGridSize[2]);
   	printf("Current grid size: %dx%dx%d\n",myDimGrid.x,myDimGrid.y,myDimGrid.z);
	if ( (myDimGrid.x<=x.maxGridSize[0])&&(myDimGrid.y<=x.maxGridSize[1])&&(myDimGrid.z<=x.maxGridSize[2]) ) {
		printf("The GPU can support this grid size\n");
	} else {
		printf("The GPU can NOT support this grid size\n");
		exit(1);
	}
}
