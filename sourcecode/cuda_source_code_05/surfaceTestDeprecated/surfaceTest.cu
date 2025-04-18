#include <stdio.h>
#include <string.h>

// 2D surfaces 
surface<void, 2> inputSurface; 
surface<void, 2> outputSurface; 

// Simple copy kernel 
__global__ void copyKernel(int width, int height) { 
	// Calculate surface coordinates 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x < width && y < height) { 
		uchar4 data; 
		// Read from input surface 
		surf2Dread(&data, inputSurface, x * 4, y); 
		// Write to output surface 
		surf2Dwrite(data, outputSurface, x * 4, y); 
	} 
}

__global__ void transformTextureToGlobal ( float* gpu_odata, int width, int height) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ( (x < width) && (y < height) ) {
		// Read from input surface, save into the Global memory
		surf2Dread(&(gpu_odata[y*width+x]), outputSurface, x*4 , y); 
	}
}

void surfTest( int argc, char** argv) { 
	unsigned int ui;
	int width = 5;
	int height = 5;
	int sizeInt = width * height;
	int size = width * height * sizeof(float);
	float *host_input_data = (float*) malloc(size);

	for (ui=0;ui<sizeInt;ui++) {
		host_input_data[ui]=(float)(ui+0.1);
	}

	// Allocate CUDA arrays in device memory 
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); 
	cudaArray* cuInputArray; 
	cudaMallocArray(&cuInputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore); 
	cudaArray* cuOutputArray; 
	cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore); 
	// Copy to device memory some data located at address host_input_data 
	// in host memory 
	cudaMemcpyToArray(cuInputArray, 0, 0, host_input_data, size, cudaMemcpyHostToDevice); 
	// Bind the arrays to the surface references 
	cudaBindSurfaceToArray(inputSurface, cuInputArray); 
	cudaBindSurfaceToArray(outputSurface, cuOutputArray); 

	// Invoke kernel 
	dim3 dimBlock(5, 5, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
	//dim3 dimBlock(16, 16);
	//dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,(height + dimBlock.y - 1) / dimBlock.y);
	copyKernel<<<dimGrid, dimBlock>>>(width, height); 

	// allocate mem for the result on host side
	float* host_output_data = (float*) malloc( size );

	// allocate device memory for result
	float* device_output_data = NULL;
	cudaMalloc( (void**) &device_output_data, size);

	transformTextureToGlobal <<<dimGrid, dimBlock>>> (device_output_data,width, height); 

	// Copy from the gpu the result
	cudaMemcpy( host_output_data, device_output_data, size, cudaMemcpyDeviceToHost);
	printf("host_output_data post copy, if the surface memory worked, should be the same as host_input_data: \n");
	for (ui=0;ui<sizeInt;ui++) {
		printf("host_input_data[%d]=%f host_output_data[%d]=%f\n",ui,host_input_data[ui],ui,host_output_data[ui]);
	}

	// Free device memory 
	cudaFreeArray(cuInputArray); 
	cudaFreeArray(cuOutputArray); 
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
