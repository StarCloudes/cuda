//=============================================================================================
// Name        		: thrustReduce.cu
// Author      		: Jose Refojo
// Version     		:	14-02-2014
// Creation date	:	27-02-2013
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

int main(void) {
	unsigned int ui;
	long int thrustSum, stlSum;
	struct timeval stlStart, stlEnd;
	struct timeval thrustStart, thrustEnd;

	// generate 128M random numbers on the host
	thrust::host_vector<long int> h_vec(32 << 22);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);

	std::vector<long int> stl_vector;
	stl_vector.resize(h_vec.size());
	for (ui=0;ui<stl_vector.size();ui++) {
		stl_vector[ui]=h_vec[ui];
	}

	// transfer data to the device
	thrust::device_vector<long int> d_vec = h_vec;

	// reduce data on the device
	gettimeofday(&thrustStart, NULL);
	thrustSum = thrust::reduce(d_vec.begin(), d_vec.end(), (long int) 0, thrust::plus<long int>());
	// This would be how to reduce to the maximum
	//thrustSum = thrust::reduce(d_vec.begin(), d_vec.end(), (long int) 0, thrust::maximum<long int>());
	gettimeofday(&thrustEnd, NULL);

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

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	unsigned int printIterations = min(10,(int)(h_vec.size()));	
	cout << "h_vec = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << h_vec[ui] << " ";
	}
	cout << "  thrustSum = " << thrustSum << endl;
	cout << "stl_vector = ";
	for (ui=0;ui<printIterations;ui++) {
		cout << stl_vector[ui] << " ";
	}
	cout << "  stlSum = " << stlSum << endl;

	float thrustTotal = (float)((thrustEnd.tv_sec * 1000000 + thrustEnd.tv_usec) - (thrustStart.tv_sec * 1000000 + thrustStart.tv_usec));
	float stlTotal = (float)((stlEnd.tv_sec * 1000000 + stlEnd.tv_usec) - (stlStart.tv_sec * 1000000 + stlStart.tv_usec));
	double speedup = (double)(stlTotal)/(double)(thrustTotal);
	cout << "Total time on the GPU, with thrust: " << thrustTotal <<
		" total time on the CPU, with stl: " << stlTotal <<
		" speedup: " << speedup << endl; 

	return 0;
}
