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

using namespace std;

float	exponentialIntegralFloat		(const int n,const float x);
double	exponentialIntegralDouble		(const int n,const double x);
void	outputResultsCpu			(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int		parseArguments				(int argc, char **argv);
void	printUsage				(void);


bool verbose,timing,cpu;
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use

int main(int argc, char *argv[]) {
	unsigned int ui,uj;
	cpu=true;
	verbose=false;
	timing=false;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
	n=10;
	numberOfSamples=10;
	a=0.0;
	b=10.0;

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

	if (timing) {
		if (cpu) {
			printf ("calculating the exponentials on the cpu took: %f seconds\n",timeTotalCpu);
		}
	}

	if (verbose) {
		if (cpu) {
			outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
		}
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

	while ((c = getopt (argc, argv, "cghn:m:a:b:tv")) != -1) {
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
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
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
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("     \n");
}
