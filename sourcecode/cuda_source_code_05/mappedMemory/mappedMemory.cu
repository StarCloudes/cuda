// mappedMemory.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
 
// define the problem and block size
#define NUMBER_OF_ARRAY_ELEMENTS 50
#define N_THREADS_PER_BLOCK 256
 
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
	float *a_mapped_device; // pointer to mapped device memory
	float *check_host;   // pointer to host memory used to check results
	int i, N = NUMBER_OF_ARRAY_ELEMENTS;
	size_t size = N*sizeof(float);
	cudaDeviceProp deviceProp;
 
#if CUDART_VERSION < 2020
#error "This CUDART version does not support mapped memory!\n"
#endif
 
	// Get properties and verify device 0 supports mapped memory
	cudaGetDeviceProperties(&deviceProp, 0);
	checkCUDAError("cudaGetDeviceProperties");
	if(!deviceProp.canMapHostMemory) {
		fprintf(stderr, "Device %d cannot map host memory!\n", 0);
		exit(EXIT_FAILURE);
	}
 
	// set the device flags for mapping host memory
	cudaSetDeviceFlags(cudaDeviceMapHost);
	checkCUDAError("cudaSetDeviceFlags");
 
	// allocate mapped arrays 
	cudaHostAlloc((void **)&a_host, size, cudaHostAllocMapped);
	checkCUDAError("cudaHostAllocMapped");
 
	// Get the device pointers to the mapped memory
	cudaHostGetDevicePointer((void **)&a_mapped_device, (void *)a_host, 0);
	checkCUDAError("cudaHostGetDevicePointer");
	// So now "a_mapped_device" and "a_host" are just mappings of the same memory on the host
 
	// initialization of host data
	for (i=0; i<N; i++) a_host[i] = (float)i;
 
	// do calculation on device:
	// Part 1 of 2. Compute execution configuration
	int blockSize = N_THREADS_PER_BLOCK;
	int nBlocks = N/blockSize + (N%blockSize > 0?1:0);
 
	// Part 2 of 2. Call incrementArrayOnDevice kernel 
	incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_mapped_device, N);
	checkCUDAError("incrementArrayOnDevice");
 
	// Note the allocation, initialization and call to incrementArrayOnHost
	// occurs asynchronously to the GPU
	check_host = (float *)malloc(size);
	for (i=0; i<N; i++) check_host[i] = (float)i;
	incrementArrayOnHost(check_host, N);
 
	// Make certain that all threads are idle before proceeding
	// TODO: Update to cudaDeviceSynchronize and test
	cudaThreadSynchronize();
	checkCUDAError("cudaThreadSynchronize");

	// check results
	printf("If the host memory worked correctly, all these values should be the same:\n");
	for (i=0; i<N; i++) {
		assert(check_host[i] == a_host[i]);
		printf("check_host[%d] = %f  a_host[%d] = %f\n",i,check_host[i],i,a_host[i]);
	}
 
	// cleanup
	free(check_host); // free host memory
	cudaFreeHost(a_host); // free mapped memory (and device pointers)
}
