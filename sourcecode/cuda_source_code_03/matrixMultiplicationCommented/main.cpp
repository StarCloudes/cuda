//=============================================================================================
// Name        		: matrixMultiplication.cu
// Author      		: Jose Refojo
// Version     		:	08-02-2018
// Creation date	:	22-09-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will multiply two matrices into a third one (AB=C)
//					  main will call the functions matmulCPU (which will be stated here) and matmulGPU (that will be stated in a different .cu file)
//					  matmulGPU allocates and transfers the matrices in the global memory of the gpu, and then sets up the kernel and runs it.
//					  This part of the code will also measure the timings
//=============================================================================================

#include <stdlib.h>
#include "stdio.h"
#include "time.h"
#include <getopt.h>

// Theses values usually go into defines so the compiler can check shared memory allocation (won't be used this time)
#define BLOCK_SIZE_X 8		// Default blocksize in the x dimension, since we are using a 2d grid
#define BLOCK_SIZE_Y 128	// Default blocksize in the y dimension, since we are using a 2d grid

// We define those variables as global - another option would be to pass them as arguments to the appropriate funcions
int MATRIX_SIZE_N = 2000;	// Those are the default values of N and M, the sizes of the matrices
int MATRIX_SIZE_M = 2000;
int block_size_x;
int block_size_y;
bool verbose = false;		// We can turn on verbose output, to show more information about the card and such
int skipCpuTest = 0;		// Since the CPU test might take quite a long time, we give an option to skip it

// Declare the functions here, so they can be described after the functions that call them
void	matmulCPU							(int N,int M,float **A,float **B,float **C);
int		parseArguments						(int argc, char **argv);						// Parse the arguments
void	printUsage							(void);											// Print the usage of the program

// This functions serves as a bridge between the main and the GPU code -  they are written in .cu cuda source code files
// By declaring them as extern, this file gets compiled and then it is up to the linker to set them up when creating the executable
extern int		chooseCudaCard				(bool verbose);									// How to pick the best available card
extern void		matmulGPU					(int N,int M,float *A1d,float *B1d,float *C1d);

int main(int argc, char *argv[]) {
	int i,j;

	// Set the default values
	block_size_x=BLOCK_SIZE_X;
	block_size_y=BLOCK_SIZE_Y;

	// Modify them if the user has specified different ones
	parseArguments(argc, argv);

	int N = MATRIX_SIZE_N;
	int M = MATRIX_SIZE_M;

	// Choose the best cuda card available
	chooseCudaCard(verbose);

	// Matrix A
	float *A1d;
	float **A;
	A1d = (float*) malloc( N*M*sizeof(float) );
	A = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		A[i]=(&(A1d[i*M]));
	}

	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<M;j++) {
			A[i][j] = i+j+1.0;
		}
	}

	// Matrix B
	float *B1d;
	float **B;
	B1d = (float*) malloc( M*N*sizeof(float) );
	B = (float**) malloc(M*sizeof(float*));
	for (i=0;i<M;i++) {
		B[i]=(&(B1d[i*N]));
	}
	// Initialization
	for (i=0;i<M;i++) {
		for (j=0;j<N;j++) {
			B[i][j] = (j+1);
		}
	}


	// Matrix C
	float *C1d;
	float *C1dGPU;
	float **C;
	C1d = (float*) malloc( N*N*sizeof(float) );
	C = (float**) malloc(N*sizeof(float*));
	for (i=0;i<N;i++) {
		C[i]=(&(C1d[i*N]));
	}
	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j] = 0;
		}
	}

	float CPUTime,GPUTime;

	printf("***********************************************************************************************\n");
	printf("******** This program will calculate the multiplication of two hard coded matrices         ****\n");
	printf("******** of sizes: A= %d x %d ; B= %d x %d; block size= %d x %d       ****\n",N,M,M,N,block_size_x,block_size_y);
	printf("***********************************************************************************************\n");

	clock_t matmulCPUStart = clock();

	if (!skipCpuTest) matmulCPU(N,M,A,B,C);	// Compute the matrix multiplication in the CPU, unless the user requested otherwise

	CPUTime=(float)(clock()-matmulCPUStart)/(float)(CLOCKS_PER_SEC);

	if (verbose) {
		printf("Matrix A:\n");
		for (i=0;i<N;i++) {
			for (j=0;j<M;j++) {
				printf("%f ",A[i][j]);
			}
			printf("\n");
		}
		printf("\n");

		printf("Matrix B:\n");
		for (i=0;i<M;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",B[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("******************************    matmulCPU    ***************************************\n");
	printf("Matrix C (calculated in the CPU):\n"); 
	if (verbose) {
		for (i=0;i<N;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",C[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	} else {
		printf(" C[%d][%d]=%f ",0,0,C[0][0]);
		printf(" C[%d][%d]=%f ",N-1,0,C[N-1][0]);
		printf(" C[%d][%d]=%f ",0,N-1,C[0][N-1]);
		printf(" C[%d][%d]=%f\n",N-1,N-1,C[N-1][N-1]);
	}

	printf("matrix multiplication on the CPU took: %f seconds\n",CPUTime);

	// Initialization
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j] = 0;
		}
	}

	clock_t matmulGPUStart = clock();

	matmulGPU (N,M,A1d,B1d,C1d);

	GPUTime=(float)(clock()-matmulGPUStart)/(float)(CLOCKS_PER_SEC);

	printf("******************************    matmulGPU    ***************************************\n");
	printf("Matrix C (calculated in the GPU):\n"); 
	if (verbose) {
		for (i=0;i<N;i++) {
			for (j=0;j<N;j++) {
				printf("%f ",C[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	} else {
		printf(" C[%d][%d]=%f ",0,0,C[0][0]);
		printf(" C[%d][%d]=%f ",N-1,0,C[N-1][0]);
		printf(" C[%d][%d]=%f ",0,N-1,C[0][N-1]);
		printf(" C[%d][%d]=%f\n",N-1,N-1,C[N-1][N-1]);
	}

	printf("matrix multiplication on the GPU took: %f seconds\n",GPUTime);
	
	printf("******************************    speedup: %f    ***************************************\n",(CPUTime/GPUTime));

	free(A);
	free(A1d);

	free(B);
	free(B1d);

	free(C);
	free(C1d);
}

void matmulCPU (int N,int M,float **A,float **B,float **C) {
	int i,j,k;
	for (i=0;i<N;i++) {
		for (j=0;j<N;j++) {
			C[i][j]=0.0f;
			for (k=0;k<M;k++) {				
				C[i][j]+=A[i][k]*B[k][j];
			}
		}
	}
}

int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "hn:m:svx:y:")) != -1) {
		switch(c) {
			case 'h':
				printUsage(); break;
			case 'n':
				 MATRIX_SIZE_N = atoi(optarg); break;
			case 'm':
				 MATRIX_SIZE_M = atoi(optarg); break;
			case 's':
				skipCpuTest = 1; break;
			case 'v':
				verbose = true; break;
			case 'x':
				 block_size_x = atoi(optarg); break;
			case 'y':
				 block_size_y = atoi(optarg); break;
			default:
				fprintf(stderr, "Invalid option given\n");
				return -1;
		}	
	}
	return 0;
}

void printUsage () {
	printf("=============================================================================================\n");
	printf(" Name                 : matrixMultiplicationCommented.cu\n");
	printf(" Author               : Jose Mauricio Refojo <jose@tchpc.tcd.ie>\n");
	printf(" Version              : 1.01\n");
	printf(" Creation date        :	01-02-2013\n");
	printf(" Current version date :	08-02-2018\n");
	printf(" Copyright            : Copyright belongs to Trinity Centre for High Performance Computing\n");
	printf(" Description          : This program will calculate the product matrix of two matrices of sizes\n");
	printf("                        nxm and mxn, by using the global memory in the GPU\n");
	printf("usage:\n");
	printf("matrixMultiplicationTexture [options]\n");
	printf("      -h           : will show this usage\n");
	printf("      -n   size    : will set n, the number of rows of the first matrix to size (default: %d)\n",MATRIX_SIZE_N);
	printf("      -m   size    : will set m, the number of columns of the first matrix to size (default: %d)\n",MATRIX_SIZE_M);
	printf("      -s           : will skip the CPU test\n");
	printf("      -v           : will run in verbose mode\n");
	printf("      -x   size    : will set the number of threads per block in the first dimension of the C matrix to size (default: %d)\n",BLOCK_SIZE_X);
	printf("      -y   size    : will set the number of threads per block in the second dimension of the C matrix to size (default: %d)\n",BLOCK_SIZE_Y);
	printf("=============================================================================================");
	printf("     \n");
}
