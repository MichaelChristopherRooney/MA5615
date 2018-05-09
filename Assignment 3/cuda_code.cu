#include <stdio.h>
#include <iostream>
#include <string>

void custom_error_check(cudaError result, std::string err_str){
	if(result != cudaSuccess){
		std::cout << err_str << "\n";
		std::cout << "Error code: " << result << "\n";
		exit(1);
	}	
}

// Taken from provided sample code
int find_best_device() {
	int i,n,best,bestNumberOfMultiprocessors;
	cudaGetDeviceCount(&n);
	int numberOfCUDAcoresForThisCC=0;
	struct cudaDeviceProp x;
	best=-1;
	bestNumberOfMultiprocessors=-1;
	for (i=0;i<n;i++) {
		cudaGetDeviceProperties(&x, i);
		switch (x.major) {
			case 1:	// Tesla / T10
				numberOfCUDAcoresForThisCC=8;
				break;
			case 2:	// Fermi
				numberOfCUDAcoresForThisCC=32;
				break;
			case 3:	// Kepler
				numberOfCUDAcoresForThisCC=192;
				break;
			case 5:	// Maxwell
				numberOfCUDAcoresForThisCC=128;
				break;
			case 6:	// Pascal
				switch (x.minor) {
					case 0: // GP100, 64 cuda cores per SM - 7.0 should be prefered over 7.1
						numberOfCUDAcoresForThisCC=64;
						break;
					case 1: // GP102, GP104, GP106, GP107, 128 cuda cores per SM
						numberOfCUDAcoresForThisCC=128;
						break;
					default: // Unknown
						numberOfCUDAcoresForThisCC=0;
						break;
				}
				numberOfCUDAcoresForThisCC=128;
				break;
			case 7:	// Volta
				numberOfCUDAcoresForThisCC=64;
				break;
			default: // Unknown
				numberOfCUDAcoresForThisCC=0;
				break;
		}
		if (x.multiProcessorCount>bestNumberOfMultiprocessors*numberOfCUDAcoresForThisCC) {
			best=i;
			bestNumberOfMultiprocessors=x.multiProcessorCount*numberOfCUDAcoresForThisCC;
		}
	}
	return best;
}

__device__ float device_exp_integral_float(int n, const float x){
	const int max_iter = 2000000000;
	const float e_const_float = 0.5772156649015329f;
	const float epsilon_float = 1.E-30f;
	int i, ii, nm1 = n-1;
	float a, b, c, d, del, fact, h, psi, ans=0.0;
	if(n == 0){
		return expf(-x)/x;
	}
	if(x > 1.0f){
		b = x + n;
		c = 3.402823E38;
		d = 1.0f/b;
		h=d;
		for(i = 1; i <= max_iter; i++){
			a=-i*(nm1+i);
			b+=2.0;
			d=1.0/(a*d+b);
			c=b+a/c;
			del=c*d;
			h*=del;
			if (fabsf(del-1.0)<=epsilon_float) {
				return h*expf(-x);
			}
		}
                return h*expf(-x);
	} else {
		ans=(nm1!=0 ? 1.0/nm1 : -log(x)-e_const_float); // First term
		fact=1.0;
		for (i=1;i<=max_iter;i++) {
			fact*=-x/i;
			if (i != nm1) {
				del = -fact/(i-nm1);
			} else {
				psi = -e_const_float;
				for (ii=1;ii<=nm1;ii++) {
					psi += 1.0/ii;
				}
				del=fact*(-log(x)+psi);
			}
			ans+=del;
			if (fabsf(del)<fabsf(ans)*epsilon_float) return ans;
		}
		return ans;
	}
}

// Note: need to use j-1 as an index since j starts at 1 rather than 0
// Same with using idx rather than my_n, as the minimum my_n is 1 rather than 0.
// TODO: move variables (like division) to constant memory to save registers
__global__ void device_part_float(
		const float division, const int n, const int num_samples, const float a, 
		float *device_float_results
	){
	const int idx=blockIdx.x*blockDim.x+threadIdx.x;
	const int my_n = idx + 1;
	if(my_n > n){
		return;
	}
	const int offset = (idx * num_samples);
	float x;
	float4 f_res;
	int j;
	for(j = 1; j <= num_samples - 4; j = j + 4){
		x = a+(j*division);
		f_res.x = device_exp_integral_float(my_n, x);
		x = a+((j+1)*division);
		f_res.y = device_exp_integral_float(my_n, x);
		x = a+((j+2)*division);
		f_res.z = device_exp_integral_float(my_n, x);
		x = a+((j+3)*division);
		f_res.w = device_exp_integral_float(my_n, x);
		*((float4 *)&(device_float_results[offset + (j-1)])) = f_res;
	}
	// Handle any remaining work if num_samples does not divide evenly by 4
	for(; j <= num_samples; j++){
                x = a+(j*division);
                f_res.x = device_exp_integral_float(my_n, x);
		device_float_results[offset + (j-1)] = f_res.x;
	}
}

__device__ double device_exp_integral_double(int n, const double x){
	const int max_iter = 2000000000;
	const double e_const_double = 0.5772156649015329;
	const double epsilon_double = 1.E-30;
	int i, ii, nm1 = n-1;
	double a, b, c, d, del, fact, h, psi, ans=0.0;
	if(n == 0){
		return exp(-x)/x;
	}
	if(x > 1.0f){
		b = x + n;
		c = 3.402823E38;
		d = 1.0f/b;
		h=d;
		for(i = 1; i <= max_iter; i++){
			a=-i*(nm1+i);
			b+=2.0;
			d=1.0/(a*d+b);
			c=b+a/c;
			del=c*d;
			h*=del;
			if (fabs(del-1.0)<=epsilon_double) {
				return h*exp(-x);
			}
		}
                return h*exp(-x);
	} else {
		ans=(nm1!=0 ? 1.0/nm1 : -log(x)-e_const_double); // First term
		fact=1.0;
		for (i=1;i<=max_iter;i++) {
			fact*=-x/i;
			if (i != nm1) {
				del = -fact/(i-nm1);
			} else {
				psi = -e_const_double;
				for (ii=1;ii<=nm1;ii++) {
					psi += 1.0/ii;
				}
				del=fact*(-log(x)+psi);
			}
			ans+=del;
			if (fabs(del)<fabs(ans)*epsilon_double) return ans;
		}
		return ans;
	}
}

// Note: need to use j-1 as an index since j starts at 1 rather than 0
// Same with using idx rather than my_n, as the minimum my_n is 1 rather than 0.
// TODO: move variables (like division) to constant memory to save registers
__global__ void device_part_double(
		const double division, const int n, const int num_samples, const double a, 
		double *device_double_results
	){
	const int idx=blockIdx.x*blockDim.x+threadIdx.x;
	const int my_n = idx + 1;
	if(my_n > n){
		return;
	}
	const int offset = (idx * num_samples);
	double x;
	double4 d_res;
	int j;
	for(j = 1; j <= num_samples - 4; j = j + 4){
		x = a+(j*division);
		d_res.x = device_exp_integral_double(my_n, x);
		x = a+((j+1)*division);
		d_res.y = device_exp_integral_double(my_n, x);
		x = a+((j+2)*division);
		d_res.z = device_exp_integral_double(my_n, x);
		x = a+((j+3)*division);
		d_res.w = device_exp_integral_double(my_n, x);
		*((double4 *)&(device_double_results[offset + (j-1)])) = d_res;
	}
	// Handle any remaining work if num_samples does not divide evenly by 4
	for(; j <= num_samples; j++){
                x = a+(j*division);
                d_res.x = device_exp_integral_double(my_n, x);
		device_double_results[offset + (j-1)] = d_res.x;
	}
}

// TODO: fix up events and record to a struct rather than printing info here
// Assuming this is run on CUDA01 it does the following:
//	1) The float code is run on the GTX 780
//	2) The double code is run on the Tesla K40c
extern void do_cuda_part(
		double a, double b, unsigned int n, unsigned int num_samples, 
		int block_size, float **float_results, double **double_results
	){
	// Data needed by both kernels
	unsigned int size = n * num_samples;
	double division = (b-a)/(double)num_samples;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (n/dimBlock.x) + (!(n%dimBlock.x)?0:1) );
	float t = 0;

	// Set up device for double version and device for float version
	int double_device_id = find_best_device();
	int float_device_id = double_device_id == 0 ? 1 : 0;

	// Set up events
	cudaSetDevice(float_device_id);
	cudaEvent_t f_start, f_stop;
	cudaEventCreate(&f_start);
	cudaEventCreate(&f_stop);
	cudaSetDevice(double_device_id);
	cudaEvent_t d_start, d_stop;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	// Set up streams
	cudaSetDevice(float_device_id);
	cudaStream_t float_stream;
	cudaStreamCreate(&float_stream);
	cudaSetDevice(double_device_id);
	cudaStream_t double_stream;
	cudaStreamCreate(&double_stream);

	// Allocate space on float device for results
	float *device_float_results;
	cudaSetDevice(float_device_id);
	cudaEventRecord(f_start, float_stream);
	custom_error_check(
		cudaMalloc((void **) &device_float_results, size * sizeof(float)), 
		"Failed to allocate float result buffer on device."
	);
	cudaEventRecord(f_stop, float_stream);
	cudaEventSynchronize(f_stop);
	cudaEventElapsedTime(&t, f_start, f_stop);
	printf("Float alloc: %f\n", t);

	// Allocate space on double device for results
	double *device_double_results;
	cudaSetDevice(double_device_id);
	cudaEventRecord(d_start, double_stream);
	custom_error_check(
		cudaMalloc((void **) &device_double_results, size * sizeof(double)), 
		"Failed to allocate double result buffer on device."
	);
	cudaEventRecord(d_stop, double_stream);
	cudaEventSynchronize(d_stop);
	cudaEventElapsedTime(&t, d_start, d_stop);
	printf("Double alloc: %f\n", t);

	// Now run the kernels on different streams
	cudaSetDevice(float_device_id);
	cudaEventRecord(f_start, float_stream);
	device_part_float<<<dimGrid,dimBlock, 0, float_stream>>>((float)division, n, num_samples, (float)a, device_float_results);
	cudaEventRecord(f_stop, float_stream);
	cudaSetDevice(double_device_id);
	cudaEventRecord(d_start, double_stream);
	device_part_double<<<dimGrid,dimBlock, 0, double_stream>>>(division, n, num_samples, a, device_double_results);
	cudaEventRecord(d_stop, double_stream);

	// Async copy results
	cudaSetDevice(float_device_id);
	custom_error_check(
		cudaMemcpyAsync(float_results[0], device_float_results, size * sizeof(float), cudaMemcpyDeviceToHost, float_stream), 
		"Failed to copy float results from device."
	);
	cudaSetDevice(double_device_id);
	custom_error_check(
		cudaMemcpyAsync(double_results[0], device_double_results, size * sizeof(double), cudaMemcpyDeviceToHost, double_stream), 
		"Failed to copy double results from device."
	);

	// Free device pointers
	cudaSetDevice(float_device_id);
	custom_error_check(
		cudaFree(device_float_results), 
		"Failed to free float results on device"
	);
	cudaSetDevice(double_device_id);
	custom_error_check(
		cudaFree(device_double_results), 
		"Failed to double results on device"
	);

	cudaEventSynchronize(f_stop);
	cudaEventElapsedTime(&t, f_start, f_stop);
	printf("Float kernel: %f\n", t);

	cudaEventSynchronize(d_stop);
	cudaEventElapsedTime(&t, d_start, d_stop);
	printf("Double kernel: %f\n", t);

	// Destroy streams
	cudaStreamDestroy(float_stream);
	cudaStreamDestroy(double_stream);
}
