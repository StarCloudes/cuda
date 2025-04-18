//=============================================================================================
// Name        		: peerToPeerMemoryAccess.cu
// Author      		: Jose Refojo
// Version     		:	21-03-2014
// Creation date	:	21-03-2014
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use the peer to peer memory so a device can access the memory space of another device 
//=============================================================================================

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
 
// define the problem and block size
#define NUMBER_OF_ARRAY_ELEMENTS 10
//#define NUMBER_OF_ARRAY_ELEMENTS 50
#define N_THREADS_PER_BLOCK 256

int findBestDevice ();

bool verbose=false;

// Add one to each element on the host
void incrementArrayOnHost(float *a, int N) {
	int i;
	for (i=0; i < N; i++) a[i] = a[i]+1.f;
}
 
// Add one to each element on the device
__global__ void incrementArrayOnDevice(float *a, int N) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N) a[idx] = a[idx]+1.f;
}
 
void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}                         
}
 
int main(void) {
	float *a_host; // pointer to host memory
	float *a_device; // pointer to device memory
	float *check_host;   // pointer to host memory used to check results
	int i, N = NUMBER_OF_ARRAY_ELEMENTS;
	int numberOfDevices;
	size_t size = N*sizeof(float);
 
#if CUDART_VERSION < 2020
#error "This CUDART version does not support mapped memory!\n"
#endif
 
	numberOfDevices = findBestDevice();
	printf ("numberOfDevices = %d\n", numberOfDevices);

	if (numberOfDevices<2) {
		if (numberOfDevices==0) {
			printf("You need at least 2 devices to run this program, none were found\n");
		} else if (numberOfDevices==1) {
			printf("You need at least 2 devices to run this program, only 1 was found\n");
		} else {
			printf("Error with the number of devices found\n");
		}
		exit(1);
	}

	// We will allocate memory in the first available device
	cudaSetDevice(0);
	// allocate array in the host
	a_host = (float*) malloc(N*sizeof(float));
	// Allocate array in the device
	cudaMalloc ((void **) &a_device, sizeof(float)*N);
	checkCUDAError("cudaMalloc");

	// initialization of host data
	for (i=0; i<N; i++) a_host[i] = (float)i;
 
	// Compute execution configuration
	int blockSize = N_THREADS_PER_BLOCK;
	int nBlocks = N/blockSize + (N%blockSize > 0?1:0);

	// Copy data from host memory to device memory
	cudaMemcpy(a_device, a_host, sizeof(float)*N, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpyHostToDevice");

	// increment one in device 0
	incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_device, N);
	checkCUDAError("incrementArrayOnDevice in device 0");

	// Set the second device
	cudaSetDevice(1);
	// Then, enable peer to peer access on device 0
	cudaDeviceEnablePeerAccess(0, 0);
	// increment one in device 1
	incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_device, N);
	checkCUDAError("incrementArrayOnDevice in device 1");

	// Note the allocation, initialization and call to incrementArrayOnHost
	// occurs asynchronously to the GPU
	check_host = (float *)malloc(size);
	for (i=0; i<N; i++) check_host[i] = (float)i;
	// Increment twice
	for (i=0;i<2;i++) {
		incrementArrayOnHost(check_host, N);
	}
 

	// Copy data from device memory to host memory
	cudaMemcpy(a_host, a_device, sizeof(float)*N, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpyDeviceToHost");

	// Make certain that all threads are idle before proceeding
	cudaThreadSynchronize();
	checkCUDAError("cudaThreadSynchronize");

	// check results
	printf("If the peer to peer memory access worked correctly, all these values should be the same:\n");
	for (i=0; i<N; i++) {
		assert(check_host[i] == a_host[i]);
		printf("check_host[%d] = %f  a_host[%d] = %f\n",i,check_host[i],i,a_host[i]);
	}
 
	// cleanup
	free(check_host);
	free(a_host);

	cudaFree(a_device);
}

int findBestDevice() {
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
		printf("Device %d - GPU model name: %s\n",i,x.name);
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
		printf("Choosing %s\n", x.name);
		cudaSetDevice(best);
	}
	// We return the number of devices, in case we want to use more than one
	printf("***************************************************\n");
	return (numberOfDevices);
}
