//=============================================================================================
// Name        		: matrixMultiplicationCommented.h
// Author      		: Jose Refojo
// Version     		:	22-09-2010
// Creation date	:	22-09-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will multiply two matrices into a third one (AB=C)
//					  main will call the functions matmulCPU and matmulGPU (matmulGPU could easily go into a different .cu file)
//					  matmulGPU allocates and transfers the matrices in the global memory of the gpu, and then sets up the kernel and runs it.
//					  The kernel uses a 2d grid (so it spawns a 2d set of threads), one thread per each element of the matrix C
//					  Each particular thread multiplies its row in A by its column in B and stores the obtained value in its position in C
//=============================================================================================


void		matmulGPU					(int N,int M,float *A1d,float *B1d,float *C1d);
