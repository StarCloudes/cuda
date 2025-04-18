//=============================================================================================
// Name        		: thrustAddVectorsWrapPointer.cu
// Author      		: Jose Refojo
// Version     		:	25-02-2013
// Creation date	:	25-02-2013
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will use how to cast raw pointers into thrust objects
//=============================================================================================


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <iomanip>

#include <stdio.h>
#include <stdlib.h>

using namespace std;
using std::cout;
using std::endl;

int main() {
	int N=18;
	int i;

	// pointers to host memory
	float *host_a, *host_b, *host_c;
	// pointers to device memory
	float *device_a, *device_b, *device_c;

	// Allocate arrays a, b and c on host
	host_a = (float*) malloc(N*sizeof(float));
	host_b = (float*) malloc(N*sizeof(float));
	host_c = (float*) malloc(N*sizeof(float));
	// Allocate arrays device_a, device_b and device_c on device
	cudaMalloc ((void **) &device_a, sizeof(float)*N);
	cudaMalloc ((void **) &device_b, sizeof(float)*N);
	cudaMalloc ((void **) &device_c, sizeof(float)*N);

	// Initialize arrays a and b
	for (i=0; i<N; i++) {
		host_a[i]= (float) 2*i;
		host_b[i]=-(float) i;
	}

	// Copy data from host memory to device memory
	cudaMemcpy(device_a,host_a, sizeof(float)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b,host_b, sizeof(float)*N, cudaMemcpyHostToDevice);

	// wrap raw pointer with a device_ptr 
	thrust::device_ptr<float> device_pointer_a(device_a);
	thrust::device_ptr<float> device_pointer_b(device_b);

	// Add a and b with thrust, result will be stored in b
	thrust::transform(device_pointer_a, device_pointer_a+N, device_pointer_b,device_pointer_b, thrust::plus<float>());

	// Transfer data back to host
	thrust::copy(device_pointer_b,device_pointer_b+N, host_c);

	// Output the results
	unsigned int ui;
	unsigned int printIterations = min(20,N);
	cout << "host_a = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << host_a[ui] << " ";
	}
	cout << endl;

	printIterations = min(20,N);
	cout << "host_b = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << host_b[ui] << " ";
	}
	cout << endl;

	printIterations = min(20,N);
	cout << "host_c = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << host_c[ui] << " ";
	}
	cout << endl;
	return 0;
}
