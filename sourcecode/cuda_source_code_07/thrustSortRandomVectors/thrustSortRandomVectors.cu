//=============================================================================================
// Name        		: thrustSortRandomVectors.cu
// Author      		: Jose Refojo
// Version     		:	26-02-2013
// Creation date	:	20-06-2012
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program shows how to use sort with thrust and compares its performance to the stl version
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

int main(void) {
	struct timeval stlStart, stlEnd;
	struct timeval thrustStart, thrustEnd;

	// generate 32M random numbers on the host
	thrust::host_vector<int> h_vec(32 << 20);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	std::vector<int> stl_vector;
	stl_vector.resize(h_vec.size());
	for (unsigned int ui=0;ui<stl_vector.size();ui++) {
		stl_vector[ui]=h_vec[ui];
	}

	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;

	// sort data on the device (846M keys per second on GeForce GTX 480)
	gettimeofday(&thrustStart, NULL);
	thrust::sort(d_vec.begin(), d_vec.end());
	gettimeofday(&thrustEnd, NULL);

	// sort data on the host
	gettimeofday(&stlStart, NULL);
	std::sort (stl_vector.begin(), stl_vector.end());
	gettimeofday(&stlEnd, NULL);

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	unsigned int ui;
	unsigned int printIterations = min(20,(int)(h_vec.size()));	
	cout << "h_vec = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << h_vec[ui] << " ";
	}
	cout << endl;
	cout << "stl_vector = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << stl_vector[ui] << " ";
	}
	cout << endl;

	float thrustTotal = (float)((thrustEnd.tv_sec * 1000000 + thrustEnd.tv_usec) - (thrustStart.tv_sec * 1000000 + thrustStart.tv_usec));
	float stlTotal = (float)((stlEnd.tv_sec * 1000000 + stlEnd.tv_usec) - (stlStart.tv_sec * 1000000 + stlStart.tv_usec));
	double speedup = (double)(stlTotal)/(double)(thrustTotal);
	cout << "Total time on the GPU, with thrust: " << thrustTotal <<
		" total time on the GPU, with stl: " << stlTotal <<
		" speedup: " << speedup << endl; 

	return 0;
}
