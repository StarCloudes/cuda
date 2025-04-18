
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "stdio.h"

float angle = 0.5f;    // angle to rotate image by (in radians)


// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;


// declaration, forward
void runTest( int argc, char** argv);


__global__ void transformKernel( float* g_odata, int width, int height) 
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// This would be how to use the normalised coordinates
	//float u = x / (float) width;
	//float v = y / (float) height;

	if ( (x < width) && (y < height) ) {
		g_odata[y*width+x] = tex2D(tex, x, y);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) {
	unsigned int ui, uj;
	unsigned int width, height;
	width=2;
	height=2;
	int size = width*height;

	float* floatArray1d;
	float** floatArray;


	// floatArray1d is the pointer to all the array malloc-ed in one dimension
	floatArray1d = (float*) malloc( size*sizeof(float) );
	if (floatArray1d==NULL) exit (1);
	// floatArray will be just pointers to the one dimension array
	floatArray = (float**) malloc((height)*sizeof(float*));
	if (floatArray==NULL) exit (1);
	for (ui=0;ui<height;ui++) {
		floatArray[ui]=(&(floatArray1d[ui*width]));
	}
	// The value of each position is the position in the 1d array
	printf("floatArray:\n");
	for (ui=0;ui<height;ui++) {
		for (uj=0;uj<width;uj++) {				
			floatArray[ui][uj]=(float)(ui*width+uj);
			printf("%f\n",floatArray[ui][uj]);
		}
	}

	// allocate device memory for result
	float* d_data = NULL;
	cudaMalloc( (void**) &d_data, size*sizeof(float));

	// Test: copy floatArray1d to d_data (in the GPU), then back, then print
	cudaMemcpy(d_data, floatArray1d, size*sizeof(float), cudaMemcpyHostToDevice);

	float* host_test1data = (float*) malloc( size*sizeof(float));
	// copy result from device to host

	printf("Test 1: \n");
	printf("host_output_data pre copy should be different from floatArray \n");
	for (ui=0;ui<size;ui++) {
		host_test1data[ui]=-(float)ui-1.1f;
		printf("(pre) host_test1data[%d]=%f\n", ui,host_test1data[ui]);
	}
	cudaMemcpy( host_test1data, d_data, size*sizeof(float), cudaMemcpyDeviceToHost);
	printf("host_output_data post copy, if the cudaMemcpy worked, should be the same as floatArray: \n");
	for (ui=0;ui<size;ui++) {
		printf("(post) host_test1data[%d]=%f\n", ui,host_test1data[ui]);
	}


    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cudaMallocArray( &cu_array, &channelDesc, width, height ); 
    cudaMemcpyToArray( cu_array, 0, 0, floatArray1d, size*sizeof(float), cudaMemcpyHostToDevice);

    // set texture parameters
/*
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates
*/
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	//tex.addressMode[0] = cudaAddressModeWrap;
	//tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModePoint;
	//tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; // do not normalize coordinates


    // Bind the array (cu_array), which contents are the same as of floatArray1d - to the texture
    cudaBindTextureToArray( tex, cu_array, channelDesc);

    dim3 dimBlock(2, 2, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // execute the kernel
    transformKernel<<< dimGrid, dimBlock, 0 >>>( d_data, width, height);

    // allocate mem for the result on host side
    float* host_output_data = (float*) malloc( size*sizeof(float) );
    // copy result from device to host

	printf("Test 2: \n");

	printf("host_output_data pre copy should be different from floatArray: \n");
	for (ui=0;ui<size;ui++) {
		host_output_data[ui]=-(int)ui;
		printf("host_output_data[%d]=%f\n", ui,host_output_data[ui]);
	}

	cudaMemcpy( host_output_data, d_data, size*sizeof(float), cudaMemcpyDeviceToHost);

	printf("host_output_data post copy, if the texture memory worked, should be the same as floatArray: \n");
	for (ui=0;ui<size;ui++) {
		printf("host_output_data[%d]=%f\n", ui,host_output_data[ui]);
	}
	

    // cleanup memory
    cudaFree(d_data);
    cudaFreeArray(cu_array);
    free(host_output_data);
    free(floatArray);
    free(floatArray1d);
}

