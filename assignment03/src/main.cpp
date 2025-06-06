///// Created by Jose Mauricio Refojo - 2014-04-02		Last changed: 2017-04-05
//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "exponentialIntegral_gpu.cuh"


using namespace std;

float	exponentialIntegralFloat		(const int n,const float x);
double	exponentialIntegralDouble		(const int n,const double x);
void	outputResultsCpu			(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int		parseArguments				(int argc, char **argv);
void	printUsage				(void);


bool verbose,timing,cpu;
int maxIterations;
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use
int blockSize = 256;

int main(int argc, char *argv[]) {
	unsigned int ui,uj;
	cpu=false;
	verbose=false;
	timing=false;
	bool gpu = true;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
	n=10;
	numberOfSamples=10;
	a=0.0;
	b=10.0;
	maxIterations=2000000000;

	struct timeval expoStart, expoEnd;

	parseArguments(argc, argv);

	if (verbose) {
		cout << "n=" << n << endl;
		cout << "numberOfSamples=" << numberOfSamples << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "timing=" << timing << endl;
		cout << "verbose=" << verbose << endl;
	}

	// Sanity checks
	if (a>=b) {
		cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
		return 0;
	}
	if (n<=0) {
		cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
		return 0;
	}
	if (numberOfSamples<=0) {
		cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
		return 0;
	}

	std::vector< std::vector< float  > > resultsFloatCpu;
	std::vector< std::vector< double > > resultsDoubleCpu;

	double timeTotalCpu=0.0;
	double timeTotalGpu=0.0;

	try {
		resultsFloatCpu.resize(n,vector< float >(numberOfSamples));
	} catch (std::bad_alloc const&) {
		cout << "resultsFloatCpu memory allocation fail!" << endl;	exit(1);
	}
	try {
		resultsDoubleCpu.resize(n,vector< double >(numberOfSamples));
	} catch (std::bad_alloc const&) {
		cout << "resultsDoubleCpu memory allocation fail!" << endl;	exit(1);
	}

	double x,division=(b-a)/((double)(numberOfSamples));

	// CPU calculation
	if (cpu) {
		gettimeofday(&expoStart, NULL);
		for (ui=1;ui<=n;ui++) {
			for (uj=1;uj<=numberOfSamples;uj++) {
				x=a+uj*division;
				resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat (ui,x);
				resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble (ui,x);
			}
		}
		gettimeofday(&expoEnd, NULL);
		timeTotalCpu=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}

	std::vector<float> resultsFloatGpu;
	std::vector<double> resultsDoubleGpu;

	//GPU calculation
	if (gpu) {
		resultsFloatGpu.resize(n * numberOfSamples);
		resultsDoubleGpu.resize(n * numberOfSamples);

		float* flatResultFloat = resultsFloatGpu.data();
		double* flatResultDouble = resultsDoubleGpu.data();

		float gpuTimeFloat = 0.0f;
		float gpuTimeDouble = 0.0f;

		// Call both float and double precision GPU wrappers
		exponentialIntegralFloatGPUWrapper(n, numberOfSamples, (float)a, (float)b, flatResultFloat, &gpuTimeFloat, blockSize);
		exponentialIntegralDoubleGPUWrapper(n, numberOfSamples, a, b, flatResultDouble, &gpuTimeDouble, blockSize);

		// Call both float and double precision GPU wrappers (stream-based versions)
		//exponentialIntegralFloatGPUStreamWrapper(n, numberOfSamples, (float)a, (float)b, flatResultFloat, &gpuTimeFloat, blockSize);
		//exponentialIntegralDoubleGPUStreamWrapper(n, numberOfSamples, a, b, flatResultDouble, &gpuTimeDouble, blockSize);
		timeTotalGpu = gpuTimeFloat + gpuTimeDouble;
	}


	if (timing) {
		if (cpu) {
			printf("[CPU] Execution time: %.6f seconds\n", timeTotalCpu);
		}
		if (gpu) {
			printf("[GPU] Execution time: %.6f seconds\n", timeTotalGpu);
		}
		if (cpu && timeTotalGpu > 0.0) {
			printf("[Speedup] CPU / GPU total = %.2fx\n", timeTotalCpu / timeTotalGpu);
		}
	}

	if (verbose && cpu) {
		outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
	}

	// Output GPU results if verbose and GPU was run
	if (verbose && gpu) {
		double division = (b - a) / ((double)(numberOfSamples));
		for (unsigned int i = 1; i <= n; ++i) {
			for (unsigned int j = 1; j <= numberOfSamples; ++j) {
				float valF = resultsFloatGpu[(i - 1) * numberOfSamples + (j - 1)];
				double valD = resultsDoubleGpu[(i - 1) * numberOfSamples + (j - 1)];
				double x = a + j * division;
				std::cout << "[GPU] exponentialIntegralDouble (" << i << "," << x << ")=" << valD << " ,";
				std::cout << "exponentialIntegralFloat  (" << i << "," << x << ")=" << valF << std::endl;
			}
		}
	}

	// Compare results
	if (cpu && gpu) {
		int mismatchFloat = 0;
		int mismatchDouble = 0;
		double division = (b - a) / ((double)(numberOfSamples));
	
		for (unsigned int i = 1; i <= n; ++i) {
			for (unsigned int j = 1; j <= numberOfSamples; ++j) {
				unsigned int idx = (i - 1) * numberOfSamples + (j - 1);
				float cpuF = resultsFloatCpu[i - 1][j - 1];
				float gpuF = resultsFloatGpu[idx];
				double cpuD = resultsDoubleCpu[i - 1][j - 1];
				double gpuD = resultsDoubleGpu[idx];
				double x = a + j * division;
	
				if (fabs(cpuF - gpuF) > 1e-5) {
					mismatchFloat++;
					std::cout << "[Float mismatch] (n=" << i << ", x=" << x
							  << "): CPU=" << cpuF << ", GPU=" << gpuF << std::endl;
				}
	
				if (fabs(cpuD - gpuD) > 1e-12) {
					mismatchDouble++;
					std::cout << "[Double mismatch] (n=" << i << ", x=" << x
							  << "): CPU=" << cpuD << ", GPU=" << gpuD << std::endl;
				}
			}
		}
	
		std::cout << "[Summary] Float mismatches : " << mismatchFloat << std::endl;
		std::cout << "[Summary] Double mismatches: " << mismatchDouble << std::endl;
	}

	return 0;
}

void	outputResultsCpu				(const std::vector< std::vector< float  > > &resultsFloatCpu, const std::vector< std::vector< double > > &resultsDoubleCpu) {
	unsigned int ui,uj;
	double x,division=(b-a)/((double)(numberOfSamples));

	for (ui=1;ui<=n;ui++) {
		for (uj=1;uj<=numberOfSamples;uj++) {
			x=a+uj*division;
			std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << endl;
		}
	}
}
double exponentialIntegralDouble (const int n,const double x) {
	static const double eulerConstant=0.5772156649015329;
	double epsilon=1.E-30;
	double bigDouble=std::numeric_limits<double>::max();
	int i,ii,nm1=n-1;
	double a,b,c,d,del,fact,h,psi,ans=0.0;


	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigDouble;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			ans=h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			//cout << "Series failed in exponentialIntegral" << endl;
			return ans;
		}
	}
	return ans;
}

float exponentialIntegralFloat (const int n,const float x) {
	static const float eulerConstant=0.5772156649015329;
	float epsilon=1.E-30;
	float bigfloat=std::numeric_limits<float>::max();
	int i,ii,nm1=n-1;
	float a,b,c,d,del,fact,h,psi,ans=0.0;

	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigfloat;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			ans=h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			return ans;
		}
	}
	return ans;
}


int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "cghn:m:s:a:b:tv")) != -1) {
		switch(c) {
			case 'g':
				cpu = true; break;   // enable CPU test only if -g is given
			case 'c':
				cpu = false; break;  // explicitly disable CPU test
			case 'h':
				printUsage(); exit(0); break;
			case 'i':
				maxIterations = atoi(optarg); break;
			case 'n':
				n = atoi(optarg); break;
			case 'm':
				numberOfSamples = atoi(optarg); break;
			case 'a':
				a = atof(optarg); break;
			case 'b':
				b = atof(optarg); break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			case 's':
				blockSize = atoi(optarg); break;
			default:
				fprintf(stderr, "Invalid option given\n");
				printUsage();
				return -1;
		}
	}
	return 0;
}
void printUsage () {
	printf("exponentialIntegral program\n");
	printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
	printf("This program will calculate a number of exponential integrals\n");
	printf("usage:\n");
	printf("exponentialIntegral.out [options]\n");
	printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
	printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
	printf("      -c           : will skip the CPU test\n");
	printf("      -g           : will run the CPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("     \n");
}
