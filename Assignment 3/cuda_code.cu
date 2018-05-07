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

__device__ float device_exp_integral_float(int n, const float x){
	int max_iter = 2000000000;
	float e_const = 0.5772156649015329;
	float epsilon = 1.E-30;
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
			if (fabs(del-1.0)<=epsilon) {
				return h*expf(-x);
			}
		}
                return h*expf(-x);
	} else {
		ans=(nm1!=0 ? 1.0/nm1 : -log(x)-e_const); // First term
		fact=1.0;
		for (i=1;i<=max_iter;i++) {
			fact*=-x/i;
			if (i != nm1) {
				del = -fact/(i-nm1);
			} else {
				psi = -e_const;
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

// TODO: move variables (like division) to constant memory to save registers
__global__ void device_float_part(
		float division, int n, int num_samples, float a, float *device_float_results
	){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int my_n = idx + 1;
	if(my_n > n){
		return;
	}
	float x;
	for(int j = 1; j <= num_samples; j++){
		x = a+(j*division);
		// Note: need to use j-1 as an index since j starts at 1 rather than 0
		// Same with using idx rather than my_n, as the minimum my_n is 1 rather than 0.
		device_float_results[(idx * num_samples) + (j-1)] = device_exp_integral_float(my_n, x);
	}
}

extern void do_cuda_float_part(double a, double b, unsigned int n, 
		unsigned int num_samples, int block_size, float **float_results
	){
	float division = (b-a)/(float)num_samples;
	unsigned int size = n * num_samples;
	float *device_float_results;
	custom_error_check(cudaMalloc((void **) &device_float_results, size * sizeof(float)), "Failed to allocate on device");
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (size/dimBlock.x) + (!(size%dimBlock.x)?0:1) );
	device_float_part<<<dimGrid,dimBlock>>>(division, n, num_samples, (float) a, device_float_results);
	custom_error_check(cudaMemcpy(float_results[0], device_float_results, size * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy data FROM device");
}

// Taken from provided sample code
extern void find_best_device() {
	int i,n,best,bestNumberOfMultiprocessors;
	int numberOfCUDAcoresForThisCC=0;
	struct cudaDeviceProp x;

	if ( cudaGetDeviceCount(&n)!=cudaSuccess ) {
		//printf("No CUDA-enabled devices were found\n");
	}
	//printf("Found %d CUDA-enabled devices\n",n);
	best=-1;
	bestNumberOfMultiprocessors=-1;
	for (i=0;i<n;i++) {
		cudaGetDeviceProperties(&x, i);
		//printf("========================= IDENTITY DATA ==================================\n");
		//printf("GPU model name: %s\n",x.name);
		if (x.integrated==1) {
			//printf("GPU The device is an integrated (motherboard) GPU\n");
		} else {
			//printf("GPU The device is NOT an integrated (motherboard) GPU - i.e. it is a discrete device\n");
		}
		//printf("GPU pciBusID: %d\n",x.pciBusID);
		//printf("GPU pciDeviceID: %d\n",x.pciDeviceID);
		//printf("GPU pciDomainID: %d\n",x.pciDomainID);
		if (x.tccDriver==1) {
			//printf("the device is a Tesla one using TCC driver\n");
		} else {
			//printf("the device is NOT a Tesla one using TCC driver\n");
		}
		//printf("========================= COMPUTE DATA ==================================\n");
		//printf("GPU Compute capability: %d.%d\n",x.major,x.minor);
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
		//printf("GPU Clock frequency in hertzs: %d\n",x.clockRate);
		//printf("GPU Device can concurrently copy memory and execute a kernel: %d\n",x.deviceOverlap);
		//printf("GPU number of multi-processors: %d\n",x.multiProcessorCount);
		//printf("GPU maximum number of threads per multi-processor: %d\n",x.maxThreadsPerMultiProcessor);
		//printf("GPU Maximum size of each dimension of a grid: %dx%dx%d\n",x.maxGridSize[0],x.maxGridSize[1],x.maxGridSize[2]);
		//printf("GPU Maximum size of each dimension of a block: %dx%dx%d\n",x.maxThreadsDim[0],x.maxThreadsDim[1],x.maxThreadsDim[2]);
		//printf("GPU Maximum number of threads per block: %d\n",x.maxThreadsPerBlock);
		//printf("GPU Maximum pitch in bytes allowed by memory copies: %u\n",(unsigned int)(x.memPitch));
		//printf("GPU Compute mode is: %d\n",x.computeMode);
		//printf("========================= MEMORY DATA ==================================\n");
		//printf("GPU total global memory: %zu bytes\n",(size_t)(x.totalGlobalMem));
		//printf("GPU peak memory clock frequency in kilohertz: %d bytes\n",x.memoryClockRate);
		//printf("GPU memory bus width: %d bits\n",x.memoryBusWidth);
		//printf("GPU L2 cache size: %d bytes\n",x.l2CacheSize);
		//printf("GPU 32-bit registers available per block: %d\n",x.regsPerBlock);
		//printf("GPU Shared memory available per block in bytes: %d\n",(int)(x.sharedMemPerBlock));
		//printf("GPU Alignment requirement for textures: %d\n",(int)(x.textureAlignment));
		//printf("GPU Constant memory available on device in bytes: %d\n",(int)(x.totalConstMem));
		//printf("GPU Warp size in threads: %d\n",x.warpSize);
		//printf("GPU maximum 1D texture size: %d\n",x.maxTexture1D);
		//printf("GPU maximum 2D texture size: %d %d\n",x.maxTexture2D[0],x.maxTexture2D[1]);
		//printf("GPU maximum 3D texture size: %d %d %d\n",x.maxTexture3D[0],x.maxTexture3D[1],x.maxTexture3D[2]);
		//printf("GPU maximum 1D layered texture dimensions: %d %d\n",x.maxTexture1DLayered[0],x.maxTexture1DLayered[1]);
		//printf("GPU maximum 2D layered texture dimensions: %d %d %d\n",x.maxTexture2DLayered[0],x.maxTexture2DLayered[1],x.maxTexture2DLayered[2]);
		//printf("GPU surface alignment: %d\n",(int)(x.surfaceAlignment));
		if (x.canMapHostMemory==1) {
			//printf("GPU The device can map host memory into the CUDA address space\n");
		} else {
			//printf("GPU The device can NOT map host memory into the CUDA address space\n");
		}
		if (x.ECCEnabled==1) {
			//printf("GPU memory has ECC support\n");
		} else {
			//printf("GPU memory does not have ECC support\n");
		}
		if (x.ECCEnabled==1) {
			//printf("GPU The device shares an unified address space with the host\n");
		} else {

			//printf("GPU The device DOES NOT share an unified address space with the host\n");
		}
		//printf("========================= EXECUTION DATA ==================================\n");
		if (x.concurrentKernels==1) {
			//printf("GPU Concurrent kernels are allowed\n");
		} else {
			//printf("GPU Concurrent kernels are NOT allowed\n");
		}
		if (x.kernelExecTimeoutEnabled==1) {
			//printf("GPU There is a run time limit for kernels executed in the device\n");
		} else {
			//printf("GPU There is NOT a run time limit for kernels executed in the device\n");
		}
		if (x.asyncEngineCount==1) {
			//printf("GPU The device can concurrently copy memory between host and device while executing a kernel\n");
		} else if (x.asyncEngineCount==2) {
			//printf("GPU The device can concurrently copy memory between host and device in both directions and execute a kernel at the same time\n");
		} else {
			//printf("GPU the device is NOT capable of concurrently memory copying\n");
		}
	}
	if (best>=0) {
		cudaGetDeviceProperties(&x, best);
		//printf("Choosing %s with %d multiprocessors\n", x.name,bestNumberOfMultiprocessors);
		cudaSetDevice(best);
	}
}
