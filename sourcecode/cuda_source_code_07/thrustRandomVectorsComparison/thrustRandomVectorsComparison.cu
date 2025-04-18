//=============================================================================================
// Name        		: thrustRandomVectorsComparison.cu
// Author      		: Jose Refojo
// Version     		:	05-03-2012
// Creation date	:	05-03-2012
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will compare different random number generators
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

// This is the include for the host curand
# include <curand.h>

#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

using namespace std;
using std::cout;
using std::endl;

int main(void) {
	int N = 32 << 20;

	cudaEvent_t thrustStart, thrustFinish;
	cudaEvent_t cuRandStart, cuRandFinish;
	cudaEvent_t stlStart, stlFinish;
	float thrustElapsedTime;
	float cuRandElapsedTime;
	float stlElapsedTime;

	cudaEventCreate(&thrustStart);
	cudaEventCreate(&thrustFinish);
	cudaEventCreate(&cuRandStart);
	cudaEventCreate(&cuRandFinish);
	cudaEventCreate(&stlStart);
	cudaEventCreate(&stlFinish);

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Step 1: Generate 32M random numbers on the host with thrust
	cudaEventRecord(thrustStart, 0); // We use 0 here because it is the "default" stream
	thrust::host_vector<int> thrustVector(N);
	thrust::generate(thrustVector.begin(), thrustVector.end(), rand);
	cudaEventRecord(thrustFinish, 0);

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Step 2: Generate 32M random numbers on the host with cuRand
	// pointers to host memory
	float *host_cuRandVector;
	// pointers to device memory
	float *device_cuRandVector;

	cudaEventRecord(cuRandStart, 0); // We use 0 here because it is the "default" stream

	host_cuRandVector = (float*) malloc(N*sizeof(float));
	cudaMalloc ((void **) &device_cuRandVector, sizeof(float)*N);
	// Declare and initialize the pseudo-random number generator
	curandGenerator_t generator;
	curandCreateGenerator (&generator,CURAND_RNG_PSEUDO_DEFAULT);
	// Set the seed
	curandSetPseudoRandomGeneratorSeed (generator,1234ULL);
	// Generate N random numbers from an uniform distribution
	curandGenerateUniform (generator, device_cuRandVector , N);
	// Copy data from device memory to host memory
	cudaMemcpy(host_cuRandVector, device_cuRandVector, sizeof(float)*N, cudaMemcpyDeviceToHost);

	cudaEventRecord(cuRandFinish, 0); // We use 0 here because it is the "default" stream
	//////////////////////////////////////////////////////////////////////////////////////////////
	// Step 3: same thing with stl
	cudaEventRecord(stlStart, 0); // We use 0 here because it is the "default" stream
	std::vector<float> stlVector;
	for (int i=0;i<N;i++) {
		stlVector.push_back(rand() / (float)RAND_MAX);
	}
	cudaEventRecord(stlFinish, 0);

	// Print c
	unsigned int ui;
	unsigned int printIterations = min(20,(int)(thrustVector.size()));	

	cout << "thrustVector = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << thrustVector[ui] << " ";
	}
	cout << endl;

	cout << "cuRandVector = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << host_cuRandVector[ui] << " ";
	}
	cout << endl;

	cout << "stlVector = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << stlVector[ui] << " ";
	}
	cout << endl;

	cudaEventSynchronize(thrustStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(thrustFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&thrustElapsedTime, thrustStart, thrustFinish);
	cudaEventSynchronize(cuRandStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(cuRandFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&cuRandElapsedTime, cuRandStart, cuRandFinish);
	cudaEventSynchronize(stlStart);  // This is optional, we shouldn't need it
	cudaEventSynchronize(stlFinish); // This isn't - we need to wait for the event to finish
	cudaEventElapsedTime(&stlElapsedTime, stlStart, stlFinish);

	cout << "Total time with thrust: " << thrustElapsedTime << " with cuRand: "<< cuRandElapsedTime << " on the host: " << stlElapsedTime << endl; 
	free(host_cuRandVector);
	cudaFree (device_cuRandVector);

	return 0;
}
