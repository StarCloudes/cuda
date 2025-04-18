#include <stdio.h>
#include <stdint.h>

// Set the type of the texture here
//typedef uint8_t textureType;  // How to use an integer type
typedef float textureType;  // How to use a float type

__global__ void kernel(cudaTextureObject_t tex, float* g_odata, int width, int height) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;

//	printf("idx[%d]*height+idy[%d] is: %2d, tex2D<textureType>(tex, %d, %d)  is: %f, tex2D<textureType>(tex, %d, %d)  is: %f  \n",
//			idx,idy,idx*height+idy, idx,idy, tex2D<textureType>(tex, idx, idy), idy,idx, tex2D<textureType>(tex, idy, idx));
//	printf("idx[%d]*width +idy[%d] is: %2d, tex2D<textureType>(tex, %d, %d)  is: %f, tex2D<textureType>(tex, %d, %d)  is: %f  \n",
//			 idx,idy,idx*width+idy, idx,idy, tex2D<textureType>(tex, idx, idy), idy,idx, tex2D<textureType>(tex, idy, idx));
//	printf("idy[%d]*height+idx[%d] is: %2d, tex2D<textureType>(tex, %d, %d)  is: %f, tex2D<textureType>(tex, %d, %d)  is: %f  \n",
//			idy,idx,idy*height+idx, idx,idy, tex2D<textureType>(tex, idx, idy), idy,idx, tex2D<textureType>(tex, idy, idx));
	printf("idy[%d]*width +idx[%d] is: %2d, tex2D<textureType>(tex, %d, %d)  is: %f, tex2D<textureType>(tex, %d, %d)  is: %f  \n",
			 idy,idx,idy*width+idx, idx,idy, tex2D<textureType>(tex, idx, idy), idy,idx, tex2D<textureType>(tex, idy, idx));

	if ( (idx < width) && (idy < height) ) {
//	if ( (idx < height) && (idy < width) ) {
//		tex2D<textureType>(tex, idx, idy);
//		g_odata[idy*height+idx] = tex2D<textureType>(tex, idy, idx);
//		g_odata[idx*height+idy] = idx*height+idy;
//		g_odata[idx*width+idy] = idx*width+idy;
//		g_odata[idx*width+idy] = tex2D<textureType>(tex, idy, idx);
		g_odata[idy*width+idx] = tex2D<textureType>(tex, idx, idy);

	}
}

int main(int argc, char **argv) {
	unsigned int ui;
	cudaDeviceProp gpuProperties;
	cudaGetDeviceProperties(&gpuProperties, 0);
	printf("The texturePitchAlignment in this gpu is: %lu\n", gpuProperties.texturePitchAlignment);

	cudaTextureObject_t tex;
	const int height = 4;
//	const int width = gpuProperties.texturePitchAlignment*AnIntegerValue; // As we will see later, width*sizeof(textureType) needs to be a multiplier of gpuProperties.texturePitchAlignment!
	const int width = 8; // should be able to use a different multiplier here
	printf("width: %d height: %d \n",width,height);
	const int totalSize = width*height;
	const int totalSizeInBytes = totalSize*sizeof(textureType);


	// We store the host texture data in a textureType which is in the RAM
	textureType texture_data_host[totalSizeInBytes];
	for (int i = 0; i < totalSize; i++) texture_data_host[i] = -i;

	// We allocate device memory for the output in the same way as we usually do, in the global memory
	float* results_device = NULL;
	cudaMalloc( (void**) &results_device, totalSize*sizeof(float));

	// We store the device texture data in a textureType that is in the global memory
	textureType* texture_data_device = 0;
	cudaMalloc((void**)&texture_data_device, totalSizeInBytes);

	// We copy the texture data to the global memory in the usual way
	cudaMemcpy(texture_data_device, texture_data_host, totalSizeInBytes, cudaMemcpyHostToDevice);

	// Describe the format of the texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = texture_data_device;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<textureType>();
	resDesc.res.pitch2D.pitchInBytes = width*sizeof(textureType);
	printf("resDesc.res.pitch2D.pitchInBytes is: %lu\n", resDesc.res.pitch2D.pitchInBytes);

	// Finally, bind the texture
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	// We set up the block in the usual way
	int block_size_x=4;
	int block_size_y=4;

	dim3 dimBlock(block_size_x,block_size_y);	// Set the number of threads per block
//	dim3 dimGrid ( (height/dimBlock.x) + (!(height%dimBlock.x)?0:1),(width/dimBlock.y) + (!(width%dimBlock.y)?0:1) );
	dim3 dimGrid ( (width/dimBlock.x) + (!(width%dimBlock.x)?0:1),(height/dimBlock.y) + (!(height%dimBlock.y)?0:1) );
	kernel<<<dimGrid, dimBlock>>>(tex, results_device, width,height);

	cudaDeviceSynchronize();
	printf("\n");

	// allocate mem for the result on host side
	float* results_host = (float*) malloc( totalSize*sizeof(float) );

	printf("Test 2: \n");

	printf("results_host before we overwrite it with the GPU results: \n");
	for (ui=0;ui<totalSize;ui++) {
		results_host[ui]=+2*(int)ui;
		printf("results_host[%2d]=%f\n", ui,results_host[ui]);
	}

	// copy results back from the device into host
	cudaMemcpy( results_host, results_device, totalSize*sizeof(float), cudaMemcpyDeviceToHost);

	printf("results_host post copy: \n");
	for (ui=0;ui<totalSize;ui++) {
		printf("results_host[%2d]=%f\n", ui,results_host[ui]);
	}
	

	return 0;
}
