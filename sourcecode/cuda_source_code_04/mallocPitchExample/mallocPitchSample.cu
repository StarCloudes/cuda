// This sample code will generate a 3d array and then add up together 

#include <math.h>
#include <stdio.h>

__global__ void addInX(cudaPitchedPtr array3DGPU, cudaExtent extent, float* array2DGPU,int pitchArray2DGPU) {
	char* devPtr   = (char*)array3DGPU.ptr;
	int pitch      = array3DGPU.pitch;
	int slicePitch = pitch*extent.height;

	//blockDim.x  : depth
	//threadIdx.x : height
	char* slice = devPtr + blockIdx.x * slicePitch;
	float* Xrow  = (float*)(slice + threadIdx.x * pitch);
	float* addition  = (float*)((char*)array2DGPU + blockIdx.x*pitchArray2DGPU);

	// We add together the last dimension of the 3d vector
	addition[threadIdx.x] = 0.0f;
	for(int x = 0; x<extent.width;x++){
		addition[threadIdx.x] += Xrow[x];
		Xrow[x]=2.0f;
	}

}

int main( int argc, char** argv) {
	int i,j,k;
	size_t dimX,dimY,dimZ;
	dimX=2;
	dimY=3;
	dimZ=4;
	//dimX=63;
	//dimY=500;
	//dimZ=500;

	////////////////////////////////////////////////////////////////////////////////////////////////
	// Memory allocation
	size_t sizeXYZ    	= dimX*dimY*dimZ;
	size_t sizeXYZBytes	= sizeXYZ * sizeof(float);
	size_t sizeYZ		= dimY*dimZ;
	size_t sizeYZBytes	= sizeYZ * sizeof(float);

	// arrayIn is  dimX x dimY x dimZ
	float* arrayIn;
	arrayIn = (float*)malloc(sizeXYZBytes);
	float* arrayAddedInGPU;
	arrayAddedInGPU = (float*)malloc(sizeYZBytes);
	float* arrayAddedInCPU;
	arrayAddedInCPU = (float*)malloc(sizeYZBytes);

	// initialize all arrayIn to 1s
	for(i=0; i<sizeXYZ;i++){
		arrayIn[i] = 2;
	}

	// We allocate 3d memory with cudaMalloc3D, with dimensions dimX x dimY x dimZ
	cudaPitchedPtr array3DGPU;
	cudaExtent     extent = make_cudaExtent(dimX,dimY,dimZ);
	cudaMalloc3D(&array3DGPU,extent);

	// We allocate 2d memory with cudaMallocPitch, with dimensions dimY x dimZ
	float* devPtr; //2D result. (y,z)
	size_t pitchArray2DGPU ;    // This variable will be set by cudaMallocPitch;
	cudaMallocPitch( (void**)&devPtr,&pitchArray2DGPU, (size_t)(dimY*sizeof(float)), dimZ);

	////////////////////////////////////////////////////////////////////////////////////////////////
	// Memory transfer
	// Copy arrayIn from host memory data to device memory data arrayIn => into array3DGPU
	cudaMemcpy3DParms p = {0};
	// describe the source pointer
	p.srcPtr.ptr   = arrayIn;
	p.srcPtr.pitch = dimX * sizeof(float);
	p.srcPtr.xsize = dimX;
	p.srcPtr.ysize = dimY;
	// describe the destination pointer
	p.dstPtr.ptr   = array3DGPU.ptr;
	p.dstPtr.pitch = array3DGPU.pitch;
	p.dstPtr.xsize = dimX;
	p.dstPtr.ysize = dimY;
	// describe the 3d
	p.extent.width  = dimX*sizeof(float);
	p.extent.height = dimY;
	p.extent.depth  = dimZ;
	// describe the direction of copy
	p.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Running the kernel
	int threadsPerBlock = dimY;
	int blocksPerGrid   = dimZ;
	addInX<<<blocksPerGrid,threadsPerBlock>>>(array3DGPU,extent,devPtr,pitchArray2DGPU);



	////////////////////////////////////////////////////////////////////////////////////////////////
	// Copy back and output the results

	// We do the same calculation in the CPU to be able to compare
	for(k=0;k<sizeYZ;k++){
		arrayAddedInCPU[k] =0.0f;
	}

	for(k=0;k<dimZ;k++){
		for(j=0;j<dimY;j++){
		//	arrayAddedInCPU[j] =0.f;
			for(i=0;i<dimX;i++){
				arrayAddedInCPU[j+k*dimY] += arrayIn[i];
				//arrayAddedInCPU[j+k*dimY] = -1;
			}
		}
	}

	// Copy back the 2d data into arrayAddedInGPU
	cudaMemcpy2D(arrayAddedInGPU,dimY*sizeof(float),devPtr,pitchArray2DGPU,dimY*sizeof(float),dimZ,cudaMemcpyDeviceToHost);

	// Copy arrayIn from host memory data to device memory data arrayIn => into array3DGPU
	cudaMemcpy3DParms pOut = {0};
	// describe the source pointer
	pOut.srcPtr.ptr   = array3DGPU.ptr;
	pOut.srcPtr.pitch = array3DGPU.pitch;
	pOut.srcPtr.xsize = dimX;
	pOut.srcPtr.ysize = dimY;
	// describe the destination pointer
	pOut.dstPtr.ptr   = arrayIn;
	pOut.dstPtr.pitch = dimX * sizeof(float);
	pOut.dstPtr.xsize = dimX;
	pOut.dstPtr.ysize = dimY;
	// describe the 3d
	pOut.extent.width  = dimX*sizeof(float);
	pOut.extent.height = dimY;
	pOut.extent.depth  = dimZ;
	// describe the direction of copy
	pOut.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&pOut);

	// Print the first 10 results or so
	//j=0;
	for (i=0;i<sizeYZ;i++) {
		//if (j>0)
			printf("arrayAddedInGPU[%d]=%f   arrayAddedInCPU[%d]=%f 3d arrayIn[%d]: %f (after modifying it in the GPU)\n",i,arrayAddedInGPU[i],i,arrayAddedInCPU[i],i,arrayIn[i]);
		//	j++;
		//if (j>=10)	break;
	}


	// Free the memory
	free(arrayIn);
	free(arrayAddedInGPU);
	free(arrayAddedInCPU);

	cudaFree(array3DGPU.ptr);
	cudaFree(devPtr);
};
