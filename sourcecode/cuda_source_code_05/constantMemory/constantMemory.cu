//=============================================================================================
// Name        		: constantMemory.cu
// Author      		: Jose Refojo
// Version     		:	29-06-2012
// Creation date	:	29-06-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will setup a two matrices in the constant memory,
//			  and then will multiply them (the multiplication won't be the best
//			  CUDA implementation possible)
//=============================================================================================


#include "stdio.h"
#define DIM_X 2
#define DIM_Y 3

__device__ __constant__ int     NCONST,MCONST;
__device__ __constant__ float   MatrixAConstantGPU[DIM_X][DIM_Y];
__device__ __constant__ float   MatrixBConstantGPU[DIM_Y][DIM_X];

__global__ void multiplyMatricesGPU(float *basicFloatArrayGPU) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;

	int myY = (idy+idx*MCONST)%NCONST;
	int myX = ( (idy+idx*MCONST)-myY )/NCONST;
	int i;
	//int tmp = (idy+idx*MCONST);
	
	if (myX<DIM_X) {
	   if (myY<DIM_Y){

		basicFloatArrayGPU[idy+idx*MCONST]=0.0f;
		for (i=0;i<MCONST;i++) {
			basicFloatArrayGPU[idy+idx*MCONST]+=MatrixAConstantGPU[myX][i]*MatrixBConstantGPU[i][myY];
		}
	   }	
	}
}

int main() {
	int i,j;

	int N=DIM_X;
	int M=DIM_Y;

	// Pointers for the matrix A
	float **MatrixA,**MatrixAOutput;
	float *MatrixA1d = NULL;
	float *MatrixAOutput1d = NULL;

	// Pointers for the matrix B
	float **MatrixB,**MatrixBOutput;
	float *MatrixB1d = NULL;
	float *MatrixBOutput1d = NULL;

	// Pointers for the DIM_X x DIM_X result matrices
	float **basicFloatArray,**basicFloatOutputArray;
	float *basicFloatArray1d = NULL;
	float *basicFloatOutputArray1d = NULL;
	float *basicFloatArrayGPU = NULL;


	// Allocate MatrixA on host (NxM)
	MatrixA1d	= (float*)  malloc( N*M*sizeof(float)  );
	MatrixAOutput1d = (float*)  malloc( N*M*sizeof(float)  );
	MatrixA		= (float**) malloc( N*  sizeof(float*) );
	MatrixAOutput	= (float**) malloc( N*  sizeof(float*) );
	// Those will be just pointers to the one dimension array
	for (i=0;i<N;i++) {
		MatrixA[i]=(&(MatrixA1d[i*M]));
		MatrixAOutput[i]=(&(MatrixAOutput1d[i*M]));
		for (j=0; j<M; j++) {
			MatrixA[i][j]=(float)(j+i*M);
			//printf("MatrixA[%d][%d]= %f, M=%d, i*M=%d\n",i,j,(float)(j+i*M),M,i*M);
		}
	}


	// Allocate MatrixB on host
	MatrixB1d	= (float*)  malloc( M*N*sizeof(float)  );
	MatrixBOutput1d = (float*)  malloc( M*N*sizeof(float)  );
	MatrixB		= (float**) malloc( M*  sizeof(float*) );
	MatrixBOutput	= (float**) malloc( M*  sizeof(float*) );
	// Those will be just pointers to the one dimension array
	for (i=0;i<M;i++) {
		MatrixB[i]=(&(MatrixB1d[i*N]));
		MatrixBOutput[i]=(&(MatrixBOutput1d[i*N]));
		for (j=0; j<N; j++) {
			MatrixB[i][j]=(float)(j+i*N);
		}
	}

	// Allocate basicFloatArray on host
	basicFloatArray1d = (float*) malloc( (N)*(N)*sizeof(float) );
	basicFloatOutputArray1d = (float*) malloc( (N)*(N)*sizeof(float) );
	basicFloatArray = (float**) malloc((N)*sizeof(float*));
	basicFloatOutputArray = (float**) malloc((N)*sizeof(float*));
	// Those will be just pointers to the one dimension array
	for (i=0;i<N;i++) {
		basicFloatArray[i]=(&(basicFloatArray1d[i*N]));
		basicFloatOutputArray[i]=(&(basicFloatOutputArray1d[i*N]));
		for (j=0; j<N; j++) {
			basicFloatArray[i][j]=0.0f;
		}
	}


	// Preprocessing printfs
	for (i=0; i<DIM_X; i++) {
		for (j=0; j<DIM_Y; j++) {
			//MatrixA[i][j]=(float)(i+j);
			printf("MatrixA[%d][%d]= %f\n",i,j,MatrixA[i][j]);
		}
	}
	printf("\n");
	for (i=0; i<DIM_Y; i++) {
		for (j=0; j<DIM_X; j++) {
			//MatrixB[i][j]=(float)(-i-j);
			printf("MatrixB[%d][%d]= %f\n",i,j,MatrixB[i][j]);
		}
	}
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			//basicFloatArray[i][j]=1.0f;
			printf("basicFloatArray[%d][%d]= %f\n",i,j,basicFloatArray[i][j]);
		}
	}

	// Transfer variables to the constant memory
	cudaMemcpyToSymbol(NCONST		,&N			,    sizeof(N)			);
	cudaMemcpyToSymbol(MCONST		,&M			,    sizeof(M)			);
	cudaMemcpyToSymbol("MatrixAConstantGPU"	,MatrixA1d		,    N*M*sizeof(float)		);
	cudaMemcpyToSymbol("MatrixBConstantGPU"	,MatrixB1d		,    M*N*sizeof(float)		);

	// Allocate arrays threadIdsGPU and constantIdsGPU on device
	cudaMalloc ((void **) &basicFloatArrayGPU, sizeof(float)*N*M);

	// Copy data from host memory to device memory (not needed, but this is how you do it)
	cudaMemcpy(basicFloatArrayGPU, basicFloatArray1d, sizeof(float)*N*M, cudaMemcpyHostToDevice);

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size,block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(N/dimBlock.y) + (!(N%dimBlock.y)?0:1) );

	// Scan information from the threads
	multiplyMatricesGPU<<<dimGrid,dimBlock>>>(basicFloatArrayGPU);


	cudaMemcpy(basicFloatOutputArray1d	,basicFloatArrayGPU	,sizeof(float)*N*N	,cudaMemcpyDeviceToHost);
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			printf("float test 2 basicFloatOutputArray[%d][%d]= %f\n",i,j,basicFloatOutputArray[i][j]);
		}
	}


	free(basicFloatArray);
	free(basicFloatArray1d);
	free(basicFloatOutputArray);
	free(basicFloatOutputArray1d);
	cudaFree(basicFloatArrayGPU);
}

