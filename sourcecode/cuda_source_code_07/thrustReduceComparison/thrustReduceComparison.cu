//=============================================================================================
// Name        		: thrustReduceComparison.cu
// Author      		: Jose Refojo
// Version     		:	04-03-2013
// Creation date	:	28-02-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use reduce with thrust and compares its performance to the cpu version
//=============================================================================================

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>


#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

using namespace std;
using std::cout;
using std::endl;

bool checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n",msg,cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
		return (false);
	}   
	return (true);
}


__global__ void sumAtomicSharedReduceVectorToScalarGPU	(int *vectorToReduce, int *reducedScalar, int N) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ int sdata[1];
	if (threadIdx.x==0) {
		sdata[0]=0;
	}
	__syncthreads();        // make sure all adds at one stage are done!
	if (idx<N) {
		atomicAdd(&sdata[0], vectorToReduce[idx]);
	}
	__syncthreads();        // make sure all adds at one stage are done!
	if (threadIdx.x==0) {
		atomicAdd(reducedScalar,sdata[0]);
	}
	return;
}

__global__ void sumAtomicReduceVectorToScalarGPU	(int *vectorToReduce, int *reducedScalar, int N) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if (idx<N) {
		atomicAdd(reducedScalar, vectorToReduce[idx]);
		//printf("sumAtomicReduceVectorToScalarGPU idx[%d] is adding %d\n",idx,vectorToReduce[idx]);
	}
	return;
}

__global__ void global_reduce_kernel(int * d_out, int * d_in, int N) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

//	if (myId==0)
//		printf("global_reduce_kernel idx[%d] is adding %d\n",myId,d_in[myId]);

	if (myId<N) {
		// do reduction in global mem
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				d_in[myId] += d_in[myId + s];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}
		//if (myId<30)
		//	printf("global_reduce_kernel idx[%d] is adding %d\n",myId,d_in[myId]);
		// only thread 0 writes result for this block back to global mem
		if (tid == 0) {
			d_out[blockIdx.x] = d_in[myId];
		}
	}
}

__global__ void shmem_reduce_kernel(int * d_out, const int * d_in, int N) {
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ int sdata[];
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	int tid  = threadIdx.x;

	if (myId<N) {

		// load shared mem from global mem
		sdata[tid] = d_in[myId];
		__syncthreads();            // make sure entire block is loaded!

		// do reduction in shared mem
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {

	if (myId==0)
		//printf("shared_reduce_kernel idx[%d] s=%d\n",myId,s);
			if (tid < s) {
				sdata[tid] += sdata[tid + s];
			}
			__syncthreads();        // make sure all adds at one stage are done!
		}

		// only thread 0 writes result for this block back to global mem
		if (tid == 0) {
			d_out[blockIdx.x] = sdata[0];
		}
	}
}

int main(int argc, char *argv[]) {
	unsigned int ui;
	long int problemSize;
	int block_size=1024;
	int sizeChoice=0;
	int thrustSum, stlSum;
	struct timeval stlStart, stlEnd;
	struct timeval thrustStart, thrustEnd;
	struct timeval atomicStart, atomicEnd;
	struct timeval atomicSharedStart, atomicSharedEnd;
	struct timeval globalStart, globalEnd;
	struct timeval sharedStart, sharedEnd;

	bool verbose=false;
	bool isPowerOf1024=false;

	if (argc>1) {
		sizeChoice = atoi(argv[1]);
	}

	switch (sizeChoice) {
		case 1:
			problemSize=(1 << 21);
			break;
		case 2:
			problemSize=(1 << 22);
			break;
		case 3:
			problemSize=(1 << 23);
			break;
		case 4:
			problemSize=(1 << 24);
			break;
		case 5:
			problemSize=(1 << 25);
			break;
		case 6:
			problemSize=((1 << 25)+(1<<23));
			break;
		case 10:
			problemSize=10;
			break;
		default:
			problemSize=(1 << 20);
			break;
	}
	thrust::host_vector<int> h_vec(problemSize);
	//thrust::host_vector<int> h_vec(32 << 20);
	//thrust::host_vector<int> h_vec(1 << 25);
	//thrust::host_vector<int> h_vec(1 << 20);
	//thrust::generate(h_vec.begin(), h_vec.end(), rand);

	for (ui=0;ui<h_vec.size();ui++) {
		h_vec[ui]=((int)(ui)%10);
	}
	
	int N = (int)(h_vec.size());
	if (N==(int)(1 << 20)) {
		isPowerOf1024=true;
	}

	dim3 dimBlock1d(block_size);
	dim3 dimGrid1d ( (N/dimBlock1d.x) + (!(N%dimBlock1d.x)?0:1) );

	std::vector<int> stl_vector;
	stl_vector.resize(h_vec.size());
	for (ui=0;ui<stl_vector.size();ui++) {
		stl_vector[ui]=h_vec[ui];
	}

	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;
	thrust::device_vector<int> d_atomic_vec = h_vec;

	// reduce data on the device with thrust
	gettimeofday(&thrustStart, NULL);
	thrustSum = thrust::reduce(d_vec.begin(), d_vec.end(), (int) 0, thrust::plus<int>());
	// This would be how to reduce to the maximum
	//thrustSum = thrust::reduce(d_vec.begin(), d_vec.end(), (int) 0, thrust::maximum<int>());
	gettimeofday(&thrustEnd, NULL);
	//d_vec.clear();

	// reduce data on the host
	gettimeofday(&stlStart, NULL);
	stlSum=0.0;
	for (ui=0;ui<stl_vector.size();ui++) {
		stlSum+=stl_vector[ui];
		// This would be how to reduce to the maximum
		//if (stl_vector[ui]>stlSum) {
		//	stlSum=stl_vector[ui];
		//}
	}
	gettimeofday(&stlEnd, NULL);

	// atomic reduce
	int h_atomic_reduced=0;
	int *d_atomic_reduced;
	cudaMalloc ((void **) &d_atomic_reduced, sizeof(int));
	cudaMemcpy(d_atomic_reduced, &h_atomic_reduced, sizeof(int), cudaMemcpyHostToDevice);
	int *d_atomic_vector_pointer = thrust::raw_pointer_cast(&(d_atomic_vec[0]));


	checkCUDAError("thrustReduceComparison => pre sumAtomicReduceVectorToScalarGPU");
	gettimeofday(&atomicStart, NULL);
	sumAtomicSharedReduceVectorToScalarGPU<<<dimGrid1d,dimBlock1d>>>(d_atomic_vector_pointer, d_atomic_reduced, N);
	gettimeofday(&atomicEnd, NULL);
	checkCUDAError("thrustReduceComparison => post sumAtomicReduceVectorToScalarGPU");

	// atomicShared reduce
	int h_atomicShared_reduced=0;
	int *d_atomicShared_reduced;
	cudaMalloc ((void **) &d_atomicShared_reduced, sizeof(int));
	cudaMemcpy(d_atomicShared_reduced, &h_atomicShared_reduced, sizeof(int), cudaMemcpyHostToDevice);
	int *d_atomicShared_vector_pointer = thrust::raw_pointer_cast(&(d_atomic_vec[0]));

	checkCUDAError("thrustReduceComparison => pre sumatomicSharedReduceVectorToScalarGPU");
	gettimeofday(&atomicSharedStart, NULL);
	sumAtomicSharedReduceVectorToScalarGPU<<<dimGrid1d,dimBlock1d>>>(d_atomicShared_vector_pointer, d_atomicShared_reduced, N);
	gettimeofday(&atomicSharedEnd, NULL);
	checkCUDAError("thrustReduceComparison => post sumatomicSharedReduceVectorToScalarGPU");

	// global  reduce
	int currentN=N;
	int step=1;
	int h_global_reduced=0;
	thrust::device_vector<int> d_global_intermediate_vec = h_vec;
	thrust::device_vector<int> d_global_intermediate_vec_1 = h_vec;
	int *d_global_intermediate_vector_pointer = thrust::raw_pointer_cast(&(d_global_intermediate_vec[0]));
	int *d_global_intermediate_vector_pointer_1 = thrust::raw_pointer_cast(&(d_global_intermediate_vec_1[0]));
	int *d_global_reduced;
	cudaMalloc ((void **) &d_global_reduced, sizeof(int)*N);

	if (isPowerOf1024) {
		checkCUDAError("thrustReduceComparison => pre global_reduce_kernel");
		gettimeofday(&globalStart, NULL);
		global_reduce_kernel<<<dimGrid1d,dimBlock1d>>>(d_global_intermediate_vector_pointer,d_atomic_vector_pointer, currentN);
		currentN/=1024;
		cout << "currentN = " << currentN << endl;
		while(currentN>1024) {
			dim3 dimBlockTmp(block_size);
			dim3 dimGridTmp ( (currentN/dimBlock1d.x) + (!(currentN%dimBlock1d.x)?0:1) );

			if (step%2) {
				global_reduce_kernel<<<dimGridTmp,dimBlockTmp>>>(d_global_intermediate_vector_pointer_1,d_global_intermediate_vector_pointer, currentN);
			} else {
				global_reduce_kernel<<<dimGridTmp,dimBlockTmp>>>(d_global_intermediate_vector_pointer,d_global_intermediate_vector_pointer_1, currentN);
			}
			step++;
			currentN/=1024;
			cout << "currentN = " << currentN << " step = " << step << endl;
		}
		{
			dim3 dimBlockSmall(dimGrid1d.x);
			dim3 dimGridSmall ( 1 );
			if (step%2) {
				global_reduce_kernel<<<dimGridSmall,dimBlockSmall>>>(d_global_reduced,d_global_intermediate_vector_pointer, N);
			} else {
				global_reduce_kernel<<<dimGridSmall,dimBlockSmall>>>(d_global_reduced,d_global_intermediate_vector_pointer_1, N);
			}
			gettimeofday(&globalEnd, NULL);
			checkCUDAError("thrustReduceComparison => post global_reduce_kernel");
		}
	}

	// shared reduce
	currentN=N;
	step=1;
	int h_shared_reduced=0;
	thrust::device_vector<int> d_shared_intermediate_vec = h_vec;
	thrust::device_vector<int> d_shared_intermediate_vec_1 = h_vec;
	int *d_shared_intermediate_vector_pointer = thrust::raw_pointer_cast(&(d_shared_intermediate_vec[0]));
	int *d_shared_intermediate_vector_pointer_1 = thrust::raw_pointer_cast(&(d_shared_intermediate_vec_1[0]));
	int *d_shared_reduced;
	cudaMalloc ((void **) &d_shared_reduced, sizeof(int)*N);

	if (isPowerOf1024) {
		checkCUDAError("thrustReduceComparison => pre shared_reduce_kernel");
		gettimeofday(&sharedStart, NULL);
		shmem_reduce_kernel<<<dimGrid1d,dimBlock1d,block_size * sizeof(float)>>>(d_shared_intermediate_vector_pointer,d_atomic_vector_pointer, currentN);
		currentN/=1024;
		cout << "currentN = " << currentN << endl;
		while(currentN>1024) {
			dim3 dimBlockTmp(block_size);
			dim3 dimGridTmp ( (currentN/dimBlock1d.x) + (!(currentN%dimBlock1d.x)?0:1) );

			if (step%2) {
				shmem_reduce_kernel<<<dimGridTmp,dimBlockTmp,block_size * sizeof(float)>>>(d_shared_intermediate_vector_pointer_1,d_shared_intermediate_vector_pointer, currentN);
			} else {
				shmem_reduce_kernel<<<dimGridTmp,dimBlockTmp,block_size * sizeof(float)>>>(d_shared_intermediate_vector_pointer,d_shared_intermediate_vector_pointer_1, currentN);
			}
			step++;
			currentN/=1024;
			cout << "currentN = " << currentN << " step = " << step << endl;
		}

		{
			dim3 dimBlockSmall(block_size);
			dim3 dimGridSmall ( 1 );
			if (step%2) {
				shmem_reduce_kernel<<<dimGrid1d,dimBlock1d,block_size * sizeof(float)>>>(d_shared_reduced,d_shared_intermediate_vector_pointer, N);
			} else {
				shmem_reduce_kernel<<<dimGrid1d,dimBlock1d,block_size * sizeof(float)>>>(d_shared_reduced,d_shared_intermediate_vector_pointer_1, N);
			}
			gettimeofday(&sharedEnd, NULL);
			checkCUDAError("thrustReduceComparison => post shared_reduce_kernel");
		}
	}

	// transfer data back to host
	cudaMemcpy(&h_atomic_reduced, d_atomic_reduced, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_atomicShared_reduced, d_atomicShared_reduced, sizeof(int), cudaMemcpyDeviceToHost);

	if (isPowerOf1024) {
		cudaMemcpy(&h_global_reduced, d_global_reduced, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_shared_reduced, d_shared_reduced, sizeof(int), cudaMemcpyDeviceToHost);
	}

	unsigned int printIterations = min(10,(int)(h_vec.size()));
	if (verbose) {
		cout << "h_vec = ";
		for (ui=0;ui<printIterations;ui++) {
			cout << h_vec[ui] << " ";
		cout << "stl_vector = ";
		for (ui=0;ui<printIterations;ui++) {
			cout << stl_vector[ui] << " ";
		}
	}

		}
	cout << "  thrustSum              = " << thrustSum << endl;
	cout << "  stlSum                 = " << stlSum << endl;
	cout << "  h_atomic_reduced       = " << h_atomic_reduced << endl;
	cout << "  h_atomicShared_reduced = " << h_atomicShared_reduced << endl;

	if (isPowerOf1024) {
		cout << "  h_global_reduced       = " << h_global_reduced << endl;
		cout << "  h_shared_reduced       = " << h_shared_reduced << endl;
	}

	float globalTotal,sharedTotal;
	float thrustTotal = (float)((thrustEnd.tv_sec * 1000000 + thrustEnd.tv_usec) - (thrustStart.tv_sec * 1000000 + thrustStart.tv_usec));
	float stlTotal = (float)((stlEnd.tv_sec * 1000000 + stlEnd.tv_usec) - (stlStart.tv_sec * 1000000 + stlStart.tv_usec));
	float atomicTotal = (float)((atomicEnd.tv_sec * 1000000 + atomicEnd.tv_usec) - (atomicStart.tv_sec * 1000000 + atomicStart.tv_usec));
	float atomicSharedTotal = (float)((atomicSharedEnd.tv_sec * 1000000 + atomicSharedEnd.tv_usec) - (atomicSharedStart.tv_sec * 1000000 + atomicSharedStart.tv_usec));
	if (isPowerOf1024) {
		globalTotal = (float)((globalEnd.tv_sec * 1000000 + globalEnd.tv_usec) - (globalStart.tv_sec * 1000000 + globalStart.tv_usec));
		sharedTotal = (float)((sharedEnd.tv_sec * 1000000 + sharedEnd.tv_usec) - (sharedStart.tv_sec * 1000000 + sharedStart.tv_usec));
	}

	double globalSpeedup,sharedSpeedup;
	double thrustSpeedup = (double)(stlTotal)/(double)(thrustTotal);
	double atomicSpeedup = (double)(stlTotal)/(double)(atomicTotal);
	double atomicSharedSpeedup = (double)(stlTotal)/(double)(atomicSharedTotal);
	if (isPowerOf1024) {
		globalSpeedup = (double)(stlTotal)/(double)(globalTotal);
		sharedSpeedup = (double)(stlTotal)/(double)(sharedTotal);
	}

	cout << " Total time on the CPU, with stl: " << stlTotal << endl;
	cout << " Total time on the GPU, with thrust: " << thrustTotal << " speedup for thrust: " << thrustSpeedup << endl;
	cout << " Total time on the GPU, with atomic: " << atomicTotal << " speedup for atomic: " << atomicSpeedup << endl;
	cout << " Total time on the GPU, with atomicShared: " << atomicSharedTotal << " speedup for atomicShared: " << atomicSharedSpeedup << endl;
	if (isPowerOf1024) {
		cout << " Total time on the GPU, with global: " << globalTotal << " speedup for global: " << globalSpeedup << endl;
		cout << " Total time on the GPU, with shared: " << sharedTotal << " speedup for shared: " << sharedSpeedup << endl;
	}

	cudaFree(d_atomic_reduced);
	//cudaFree(d_global_reduced);
	
	return 0;
}
