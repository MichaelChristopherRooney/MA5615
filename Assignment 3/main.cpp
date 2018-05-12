///// Created by Jose Mauricio Refojo - 2014-04-02		Last changed: 2014-04-02
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

#include "results.h"

extern void do_cuda_part(
		double a, double b, unsigned int n, unsigned int num_samples, 
		int block_size, float **float_results, double **double_results
	);

using namespace std;

float	exponentialIntegralFloat		(const int n,const float x);
double	exponentialIntegralDouble		(const int n,const double x);
void	outputResultsCpu			(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int		parseArguments				(int argc, char **argv);
void	printUsage				(void);

void output_results_cuda();
void allocate_cuda_results();
void compare_results();

bool verbose,timing,cpu,cuda; // TODO: read CUDA value from args
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use
int block_size; // TODO: read from args
float **cuda_float_results;
double **cuda_double_results;
std::vector< std::vector< float  > > resultsFloatCpu;
std::vector< std::vector< double > > resultsDoubleCpu;

struct cuda_results_s timings;

int main(int argc, char *argv[]) {
	unsigned int ui,uj;
	cpu=true;
	cuda=true;
	verbose=false;
	timing=false;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
	n=10;
	numberOfSamples=10;
	a=0.0;
	b=10;
	block_size = 256;

	struct timeval expoStart, expoEnd;

	parseArguments(argc, argv);

	if (verbose) {
		cout << "n=" << n << endl;
		cout << "numberOfSamples=" << numberOfSamples << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "timing=" << timing << endl;
		cout << "verbose=" << verbose << endl;
		cout << "block size=" << block_size << endl;
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


	double timeTotalCpu=0.0;
	double timeTotalCuda=0.0;

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
	if(cuda){
		allocate_cuda_results();
		gettimeofday(&expoStart, NULL);
		do_cuda_part(a, b, n, numberOfSamples, block_size, cuda_float_results, cuda_double_results);
		gettimeofday(&expoEnd, NULL);
		timeTotalCuda=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}
	if (timing) {
		if (cpu) {
			printf ("calculating the exponentials on the cpu took: %f seconds\n",timeTotalCpu);
		}
		if(cuda){
			printf ("calculating the exponentials with CUDA took: %f seconds\n",timeTotalCuda);
			printf("Allocating space for float results on device took: %f milliseconds\n", timings.float_alloc_time);
			printf("Allocating space for double results on device took: %f milliseconds\n", timings.double_alloc_time);
			printf("Float kernel took: %f milliseconds\n", timings.float_kernel_time);
			printf("Double kernel took: %f milliseconds\n", timings.double_kernel_time);
			printf("Copying float results from device took: %f milliseconds\n", timings.float_copy_time);
			printf("Copying double results from device took: %f milliseconds\n", timings.double_copy_time);
		}
		if(cpu && cuda){
			printf("CUDA version was %f times as fast as CPU version.\n", timeTotalCpu / timeTotalCuda);
		}
	}
	if (verbose) {
		if (cpu) {
			outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
		}
		if(cuda){
			output_results_cuda();
		}
	}
	if(cpu && cuda){
		compare_results();
	}
	return 0;
}

void compare_results(){
	unsigned int ui,uj;
        for (ui=1;ui<=n;ui++) {
       	        for (uj=1;uj<=numberOfSamples;uj++) {
			float f_cpu = resultsFloatCpu[ui-1][uj-1];
			float f_cuda = cuda_float_results[ui-1][uj-1];
			if(fabs(f_cpu-f_cuda) > 1.E-5){
				std::cout << "ERROR: float result [" << ui << ", " << uj << "] differs\n";
				return;
			}
			double d_cpu = resultsDoubleCpu[ui-1][uj-1];
			double d_cuda = cuda_double_results[ui-1][uj-1];
			if(fabs(d_cpu-d_cuda) > 1.E-5){
				std::cout << "ERROR: double result [" << ui << ", " << uj << "] differs\n";
				return;
			}
       	        }
        }
	std::cout << "SUCCESS: CPU and CUDA results match\n";
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

void output_results_cuda(){
	unsigned int ui, uj;
	double x,division=(b-a)/((double)(numberOfSamples));
        for (ui=1;ui<=n;ui++) {
                for (uj=1;uj<=numberOfSamples;uj++) {
                        x=a+uj*division;
			std::cout << "CUDA==> exponentialIntegralDouble (" << ui << "," << x <<")=" << cuda_double_results[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << cuda_float_results[ui-1][uj-1] << endl;
                }
        }
}

double exponentialIntegralDouble (const int n,const double x) {
	static const int maxIterations=2000000000;
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
			//cout << "Continued fraction failed in exponentialIntegral" << endl;
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
	static const int maxIterations=2000000000;
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

	while ((c = getopt (argc, argv, "cghn:m:a:b:tvs:")) != -1) {
		switch(c) {
			case 'c':
				cpu=false; break;	 //Skip the CPU test
			case 'h':
				printUsage(); exit(0); break;
			case 'n':
				n = atoi(optarg); break;
			case 'm':
				numberOfSamples = atoi(optarg); break;
			case 'a':
				a = atof(optarg); break;
			case 'b':
				b = atof(optarg); break;
			case 's':
				block_size = atoi(optarg); break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			case 'g':
				cuda=false; break;
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
	printf("      -g           : will skip the GPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -s           : will set the block size (default: 256)\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("     \n");
}

void allocate_cuda_results(){
	cuda_float_results = (float **)malloc(n*sizeof(float *));
	cuda_double_results = (double **)malloc(n*sizeof(double *));
	float *temp_f = (float *)malloc(numberOfSamples*n*sizeof(float));
	double *temp_d = (double *)malloc(numberOfSamples*n*sizeof(double));
	for(unsigned int i = 0; i < n; i++){
		cuda_float_results[i] = &(temp_f[i*numberOfSamples]);
		cuda_double_results[i] = &(temp_d[i*numberOfSamples]);
	}
}


