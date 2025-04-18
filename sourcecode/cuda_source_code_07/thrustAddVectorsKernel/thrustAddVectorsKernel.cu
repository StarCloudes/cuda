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

__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int Ntot) {
       int idx=blockIdx.x*blockDim.x+threadIdx.x;
       if ( idx <Ntot )
       out[idx]=in1[idx]+in2[idx];
}

int main() {
	int N=18;
	int i;

	// Allocate vectors on host memory
	thrust::host_vector<float> a_vec(N);
	thrust::host_vector<float> b_vec(N);
	thrust::host_vector<float> c_vec(N);

	// Initialize vectors 
	thrust::host_vector<float>::iterator iter;

	iter = a_vec.begin();
	i=0;
	for (iter = a_vec.begin(); iter < a_vec.end(); iter++) {
		*iter= (float) (i+1);
		i++;
	}

	iter = b_vec.begin();
	i=0;
	for (iter = b_vec.begin(); iter < b_vec.end(); iter++) {
		*iter=(float)(i);
		i++;
	}

	// Copy data from host memory to device memory 
	thrust::device_vector<float> a_gpu_vec = a_vec;
	thrust::device_vector<float> b_gpu_vec = b_vec;
	thrust::device_vector<float> c_gpu_vec = c_vec;
	// Convert the thrust::device_vector<float> to float* in device memory
	float *a_gpu_ptr = thrust::raw_pointer_cast(&(a_gpu_vec[0]));
	float *b_gpu_ptr = thrust::raw_pointer_cast(&(b_gpu_vec[0]));
	float *c_gpu_ptr = thrust::raw_pointer_cast(&(c_gpu_vec[0]));

	// Compute the execution configuration
	int block_size=8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	// Add arrays a and b, store result in c
	add_arrays_gpu<<<dimGrid,dimBlock>>>(a_gpu_ptr, b_gpu_ptr, c_gpu_ptr, N);
	// This is the same as:
	//thrust::transform(a_gpu_vec.begin(), a_gpu_vec.end(), b_gpu_vec.begin(), b_gpu_vec.begin(), thrust::plus<float>());

	// Transfer data back to host
	thrust::copy(c_gpu_vec.begin(), c_gpu_vec.end(), c_vec.begin());

	// Output the results
	unsigned int ui;
	unsigned int printIterations = min(20,(int)(a_vec.size()));
	cout << "a_vec = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << a_vec[ui] << " ";
	}
	cout << endl;

	printIterations = min(20,(int)(b_vec.size()));
	cout << "b_vec = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << b_vec[ui] << " ";
	}
	cout << endl;

	printIterations = min(20,(int)(c_vec.size()));
	cout << "c_vec = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << setw(3) << c_vec[ui] << " ";
	}
	cout << endl;
	return 0;
}
