//=============================================================================================
// Name        		: warpShuffle.cu
// Author      		: Jose Refojo
// Version     		:	05-11-2018
// Creation date	:	24-04-2014
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will run a number of warp shuffle tests
//=============================================================================================

#include "stdio.h"
#include <string>
#include <bitset>
#include <iostream>     // std::cout, std::endl
#include <iomanip>      // std::setw

#define PROBLEM_SIZE 48
#define BLOCK_SIZE 32
#define FULL_MASK 0xffffffff	// This is 32 "1"s in binary representation, so in this case, a binary mask that marks the 32 threads of a warp


bool checkCUDAError(const char *msg);
void checkCUDAErrorPeek (const char *msg);

using namespace std;
//				   0         1         2         3
//				   01234567890123456789012345678901
// 43690		is 00000000000000001010101010101010
// 2863311530	is 10101010101010101010101010101010


__global__ void warpShuffle( int *outAll, int *outAny, int *outBallot,
							 int *outInputLane, int *outInput,
//							 Uncomment to use the match functions (Only CC 7.0 or more)
//							 int *outMatchAll, int *outMatchAny,
							 int *outInputUp, int *outInputDown,
							 int Ntot) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int inputLane = -1;
	int local = -2;
	if ( idx <Ntot ) {
		local = idx;
//	}
		int allVote= __all_sync(FULL_MASK,local%2);
		outAll[idx]		= __all_sync(FULL_MASK,local%2);

		int anyVote= __any_sync(FULL_MASK,local%2);
		outAny[idx]		= __any_sync(FULL_MASK,local%2);

		int ballotVote = __ballot_sync(FULL_MASK,local%2);
		outBallot[idx]	= __ballot_sync(FULL_MASK,local%2);

//		Uncomment to use the match functions (Only CC 7.0 or more)
//		unsigned int matchAll = __match_all_sync(FULL_MASK, local,outAll);
//		outMatchAll[idx] =__match_all_sync(FULL_MASK, local,outAll);
//		unsigned int matchAny = __match_any_sync(FULL_MASK, local);
//		outMatchAny[idx] =__match_any_sync(FULL_MASK, local);

		// This is in case we want to print the results from the GPU
		//printf ("warp vote[%d], allVote=%d anyVote=%d ballotVote=%u matchAny=\n",idx,allVote,anyVote,ballotVote);

		__syncthreads();
		if (idx==0) {
			printf("\n");
		}

		int myLane = threadIdx.x%warpSize;
		int halfWarp = warpSize/2;

		if (myLane>halfWarp) {
			inputLane=myLane-halfWarp;
		} else {
			inputLane=warpSize-halfWarp-myLane;
		}
		outInputLane[idx]	=inputLane;
		// __shfl_sync fetches the variable local from the inputLane in the same warp
		outInput[idx]		=__shfl_sync(FULL_MASK,local,inputLane);
		// __shfl_up_sync fetches the variable local from the (lane-2) in the same warp, or from the same lane if (lane-2) is not valid (<0)
		outInputUp[idx]		=__shfl_up_sync(FULL_MASK,local,2);
		// __shfl_down_sync fetches the variable local from the (lane+2) in the same warp, or from the same lane if (lane+2) is not valid (>=32)
		outInputDown[idx]	=__shfl_down_sync(FULL_MASK,local,2);

		// This is in case we want to print the results from the GPU
//		printf ("warp shuffle[%2d], inputLane=%2d myLane=%2d input=%2d inputUp=%2d inputDown=%2d\n",idx,inputLane,myLane,input,inputUp,inputDown);
	}
}

int main() {
	// pointers to host memory
	int *outAllVote_h,*outAnyVote_h,*outBallotVote_h;
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	int *outMatchAll_h,*outMatchAny_h;
	int *outInputLane_h,*outInput_h,*outInputUp_h,*outInputDown_h;
	// pointers to device memory
	int *outAllVote_d,*outAnyVote_d,*outBallotVote_d;
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	int *outMatchAll_d,*outMatchAny_d;
	int *outInputLane_d,*outInput_d,*outInputUp_d,*outInputDown_d;

	int N=PROBLEM_SIZE;
	int i;

	// Allocate arrays on host
	outAllVote_h 		= (int*) malloc(N*sizeof(int));
	outAnyVote_h		= (int*) malloc(N*sizeof(int));
	outBallotVote_h		= (int*) malloc(N*sizeof(int));
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	outMatchAll_h		= (int*) malloc(N*sizeof(int));
//	outMatchAny_h		= (int*) malloc(N*sizeof(int));
	outInputLane_h		= (int*) malloc(N*sizeof(int));
	outInput_h			= (int*) malloc(N*sizeof(int));
	outInputUp_h		= (int*) malloc(N*sizeof(int));
	outInputDown_h		= (int*) malloc(N*sizeof(int));

	// Allocate arrays on device
	cudaMalloc ((void **) &outAllVote_d			, sizeof(int)*N);
	cudaMalloc ((void **) &outAnyVote_d			, sizeof(int)*N);
	cudaMalloc ((void **) &outBallotVote_d		, sizeof(int)*N);
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	cudaMalloc ((void **) &outMatchAll_d		, sizeof(int)*N);
//	cudaMalloc ((void **) &outMatchAny_d		, sizeof(int)*N);
	cudaMalloc ((void **) &outInputLane_d		, sizeof(int)*N);
	cudaMalloc ((void **) &outInput_d			, sizeof(int)*N);
	cudaMalloc ((void **) &outInputUp_d			, sizeof(int)*N);
	cudaMalloc ((void **) &outInputDown_d		, sizeof(int)*N);
	checkCUDAError("Allocate arrays on device");

	// Initialize arrays
//	for (i=0; i<N; i++) {
//		a[i]= (float) 2*i;
//	}

	// Copy data from host memory to device memory (not really needed in this case)
	cudaMemcpy(outAllVote_d,		outAllVote_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outAnyVote_d,		outAnyVote_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outBallotVote_d,		outBallotVote_h,	sizeof(int)*N, cudaMemcpyHostToDevice);
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	cudaMemcpy(outMatchAll_d,		outMatchAll_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
//	cudaMemcpy(outMatchAny_d,		outMatchAny_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outInputLane_d,		outInputLane_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outInput_d,			outInput_h,			sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outInputUp_d,		outInputUp_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(outInputDown_d,		outInputDown_h,		sizeof(int)*N, cudaMemcpyHostToDevice);
	checkCUDAError("Copy data from host memory to device memory");

	// Compute the execution configuration
	int block_size=BLOCK_SIZE;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Call the kernel
	warpShuffle<<<dimGrid,dimBlock>>>(	outAllVote_d, outAnyVote_d, outBallotVote_d,
										// Uncomment to use the match functions (Only CC 7.0 or more)
//										outMatchAll_d,outMatchAny_d,
										outInputLane_d,outInput_d,outInputUp_d,outInputDown_d, N);
	checkCUDAError("Call the kernel");

	// Copy data from device memory to host memory
	cudaMemcpy(outAllVote_h,		outAllVote_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outAnyVote_h,		outAnyVote_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outBallotVote_h,		outBallotVote_d,	sizeof(int)*N, cudaMemcpyDeviceToHost);
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	cudaMemcpy(outMatchAll_h,		outMatchAll_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(outMatchAny_h,		outMatchAny_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outInputLane_h,		outInputLane_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outInput_h,			outInput_d,			sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outInputUp_h,		outInputUp_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(outInputDown_h,		outInputDown_d,		sizeof(int)*N, cudaMemcpyDeviceToHost);
	checkCUDAError("Copy data from device memory to host memory");

	// Print results

	// Print first outAllVote_h and  outAnyVote_h
	for (i=0; i<N; i++) {
//		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
		cout << "thread id= " << std::setw(2) << i <<
				" outAllVote_h["		<< std::setw(2) <<i<<"]=" << outAllVote_h[i] <<
				" outAnyVote_h["		<< std::setw(2) <<i<<"]=" << outAnyVote_h[i] << endl;
		if ( (i>0)&&(!((i+1)%16)) ) cout << endl;
	}
	cout << endl;
	cout << endl;

	// Then, outBallotVote_h, which is the widest
	for (i=0; i<N; i++) {
//		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
		cout << "thread id= " << std::setw(2) << i <<
				" outBallotVote_h["		<< std::setw(2) <<i<<"]=" << std::bitset< 32 >(outBallotVote_h[i]) << endl;
		if ( (i>0)&&(!((i+1)%16)) ) cout << endl;
	}
	cout << endl;
	cout << endl;


	// Uncomment to use the match functions (Only CC 7.0 or more)
	// Then, outMatchAll_h and outMatchAny_h, which is the widest
//	for (i=0; i<N; i++) {
////		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
//		cout << "thread id= " << std::setw(2) << i <<
//				" outMatchAll_h["		<< std::setw(2) <<i<<"]=" << std::bitset< 32 >(outMatchAll_h[i]) <<
//				" outMatchAny_h["		<< std::setw(2) <<i<<"]=" << std::bitset< 32 >(outMatchAny_h[i]) << endl;
//		if ( (i>0)&&(!((i+1)%16)) ) cout << endl;
//	}
//	cout << endl;
//	cout << endl;

	// Then, outInputLane_h, outInput_h, outInputUp_h and outInputDown_h
	for (i=0; i<N; i++) {
//		printf(" a[%2d](%10f) + b[%2d](%10f) = c[%2d](%10f)\n",i,a[i],i,b[i],i,c[i]);
		cout << "thread id= " << std::setw(2) << i <<
				" outInputLane_h["	<< std::setw(2) <<i<<"]=" << std::setw(2) << outInputLane_h[i] <<
				" outInput_h["		<< std::setw(2) <<i<<"]=" << std::setw(2) << outInput_h[i] <<
				" outInputUp_h["	<< std::setw(2) <<i<<"]=" << std::setw(2) << outInputUp_h[i] <<
				" outInputDown_h["	<< std::setw(2) <<i<<"]=" << std::setw(2) << outInputDown_h[i] << endl;
		if ( (i>0)&&(!((i+1)%16)) ) cout << endl;
	}
	cout << endl;
	cout << endl;


	cout << "FULL_MASK =" << std::bitset< 32 >(FULL_MASK) << endl;
	cout << "43690     =" << std::bitset< 32 >(43690) << endl;
	cout << "2863311530=" << std::bitset< 32 >(2863311530) << endl;

	// 43690		is 00000000000000001010101010101010
	// 2863311530	is 10101010101010101010101010101010
//	string s = std::bitset< 32 >( 12345 ).to_string(); // string conversion
//	cout << "s (" << s << ") bitwise = >" << s << "< " << endl; // direct output
//	cout << "   " << std::bitset< 32 >( 54321 ) << ' ' << endl; // direct output

	// Free the memory on the host
	free(outAllVote_h);
	free(outAnyVote_h);
	free(outBallotVote_h);
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	free(outMatchAll_h);
//	free(outMatchAny_h);
	free(outInput_h);
	free(outInputUp_h);
	free(outInputDown_h);

	// Free the memory on the device
	cudaFree(outAllVote_d);
	cudaFree(outAnyVote_d);
	cudaFree(outBallotVote_d);
	// Uncomment to use the match functions (Only CC 7.0 or more)
//	cudaFree(outMatchAll_d);
//	cudaFree(ooutMatchAny_d);
	cudaFree(outInput_d);
	cudaFree(outInputUp_d);
	cudaFree(outInputDown_d);
}


bool checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n",msg,cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
		return (false);
	}
	return (true);
}

void checkCUDAErrorPeek (const char *msg) {
     cudaError_t err = cudaPeekAtLastError();
     if( err!= cudaSuccess ) {
          printf("(Cuda error %s): %s\n",msg,cudaGetErrorString( err) );
          exit(EXIT_FAILURE);
     }
}
