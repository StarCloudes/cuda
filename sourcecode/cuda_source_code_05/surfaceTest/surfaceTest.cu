#include <stdio.h>
#include <string.h>

// 2D surfaces 

// How it started - surface memory was in the host:
//surface<void, 2> inputSurface; 
//surface<void, 2> outputSurface; 
// How it is going - we can't do that anymore, now we have to pass them as arguments, because they are in the GPU:
//cudaSurfaceObject_t inputSurface = 0;
//cudaSurfaceObject_t outputSurface = 0;

// Simple copy kernel 
__global__ void copyKernel(cudaSurfaceObject_t inputSurface,cudaSurfaceObject_t outputSurface,int width, int height) { 
	// Calculate surface coordinates 
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
	if (idx < width && idy < height) {
		float data;

		// Read from input surface 
		surf2Dread(&data, inputSurface, idx * 4, idy); 
		// Do this printf so we can show how the data looks like fom the kernel:
		printf("x[%d]y[%d] surf2Dread(&data, inputSurface, idx * 4, idy) is: %f\n",idx,idy,data);

		// Write to output surface 
		surf2Dwrite(data, outputSurface, idx * 4, idy); 
	} 
}

__global__ void transformTextureToGlobal (cudaSurfaceObject_t outputSurface, float* gpu_odata, int width, int height) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ( (idx < width) && (idy < height) ) {
		// Read from input surface, save into the Global memory
		surf2Dread(&(gpu_odata[idy*width+idx]), outputSurface, idx*4 , idy); 
	}
}

void surfTest( int argc, char** argv) { 
	unsigned int ui;
	int width = 5;
	int height = 10;
	int sizeInt = width * height;
	int size = width * height * sizeof(float);
	float *host_input_data = (float*) malloc(size);

	// Set the input values
	for (ui=0;ui<sizeInt;ui++) {
		host_input_data[ui]=(float)(ui+0.1);
	}

	// Allocate CUDA arrays in device memory 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	cudaArray_t inputArray_Device;
	cudaMallocArray(&inputArray_Device, &channelDesc, width, height, cudaArraySurfaceLoadStore);

	cudaArray* outputArray_Device; 
	cudaMallocArray(&outputArray_Device, &channelDesc, width, height, cudaArraySurfaceLoadStore); 

	// Copy to device memory some data located at address host_input_data  in host memory 

    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
	printf("spitch = %lu\n",spitch);

//	cudaMemcpyToArray(inputArray_Device, 0, 0, host_input_data, size, cudaMemcpyHostToDevice); 	// This is deprecated
    cudaMemcpy2DToArray(inputArray_Device, 0, 0, host_input_data, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice);


    // Create the surface objects
	// This is how it was before:
	// cudaBindSurfaceToArray(inputSurface, inputArray_Device); 
	// cudaBindSurfaceToArray(outputSurface, outputArray_Device); 

	// And this is how it is now:
	// Declare the surface memory arrays	
	cudaSurfaceObject_t inputSurface = 0;
	cudaSurfaceObject_t outputSurface = 0;

    // Set up the structure for the surfaces
    struct cudaResourceDesc resDescInputSurface;
    memset(&resDescInputSurface, 0, sizeof(resDescInputSurface));
    resDescInputSurface.resType = cudaResourceTypeArray;
    resDescInputSurface.res.array.array = inputArray_Device;

    struct cudaResourceDesc resDescOutputSurface;
    memset(&resDescOutputSurface, 0, sizeof(resDescOutputSurface));
    resDescOutputSurface.resType = cudaResourceTypeArray;
    resDescOutputSurface.res.array.array = outputArray_Device;

	// Bind the arrays to the surface objects 
    cudaCreateSurfaceObject(&inputSurface, &resDescInputSurface);
    cudaCreateSurfaceObject(&outputSurface, &resDescOutputSurface);

	// Setup and Invoke kernel 
	dim3 dimBlock(4, 4, 1);
	dim3 dimGrid ( (width/dimBlock.x) + (!(width%dimBlock.x)?0:1),(height/dimBlock.y) + (!(height%dimBlock.y)?0:1) );
	copyKernel<<<dimGrid, dimBlock>>>(inputSurface,outputSurface,width, height); 

	// allocate mem for the result on host side
	float* host_output_data = (float*) malloc( size );

	// allocate device memory for result on the device
	float* device_output_data = NULL;
	cudaMalloc( (void**) &device_output_data, size);

	transformTextureToGlobal <<<dimGrid, dimBlock>>> (outputSurface,device_output_data,width, height); 

	// Copy the result from the gpu so we can print it
	cudaMemcpy( host_output_data, device_output_data, size, cudaMemcpyDeviceToHost);
	printf("host_output_data post copy, if the surface memory worked, should be the same as host_input_data: \n");
	for (ui=0;ui<sizeInt;ui++) {
		printf("host_input_data[%d]=%f host_output_data[%d]=%f\n",ui,host_input_data[ui],ui,host_output_data[ui]);
	}

	// Free device memory 
	cudaFreeArray(inputArray_Device); 
	cudaFreeArray(outputArray_Device); 
	// Free host memory
	free(host_input_data);
	free(host_output_data);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    surfTest( argc, argv);
}
