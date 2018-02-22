#include "stdlib.h"
#include "stdio.h"

#include "matrix.h"

__global__ void sum_rows(DATA_TYPE *mat, DATA_TYPE *out, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nrow){
		return;
	}
	//printf("idx: %d, blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", idx, blockIdx.x, blockDim.x, threadIdx.x);
	DATA_TYPE result = 0.0;
	for(int i = 0; i < ncol; i++){
		unsigned long long index = ((unsigned long long) ncol * (unsigned long long) idx) + (unsigned long long) i;
		//printf("Thread %d accessing %llu on iter %d\n", idx, index, i);
		result += mat[index];
	}
	out[idx] = result;
	//printf("Thread %d got %f\n", idx, result);
}

__global__ void sum_columns(DATA_TYPE *mat, DATA_TYPE *out, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= ncol){
		return;
	}
	DATA_TYPE result = 0.0;
	for(int i = 0; i < nrow; i++){
		unsigned long long index = ((unsigned long long) ncol * (unsigned long long) i) + (unsigned long long) idx;
		//printf("Thread %d accessing %d\n", idx, index);
		result += mat[index];
	}
	out[idx] = result;
	//printf("Thread %d got %f\n", idx, result);
}

extern "C" void do_gpu_col_sum(DATA_TYPE **mat, DATA_TYPE *col_sum_vec, int nrow, int ncol, int block_size){
	unsigned long long mat_size = ((unsigned long long) nrow) * ((unsigned long long) ncol) * sizeof(DATA_TYPE);
	DATA_TYPE *mat_gpu;
	DATA_TYPE *col_sum_vec_gpu;
	cudaMalloc((void **) &mat_gpu, mat_size);
	cudaMalloc((void **) &col_sum_vec_gpu, ncol * sizeof(DATA_TYPE));
	cudaMemcpy(mat_gpu, mat[0], mat_size, cudaMemcpyHostToDevice);
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (ncol/dimBlock.x) + (!(ncol%dimBlock.x)?0:1) );
	sum_columns<<<dimGrid,dimBlock>>>(mat_gpu, col_sum_vec_gpu, nrow, ncol);
	cudaMemcpy(col_sum_vec, col_sum_vec_gpu, ncol * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	cudaFree(mat_gpu);
	cudaFree(col_sum_vec_gpu);
}

extern "C" void do_gpu_row_sum(DATA_TYPE **mat, DATA_TYPE *row_sum_vec, int nrow, int ncol, int block_size){
	unsigned long long mat_size = ((unsigned long long) nrow) * ((unsigned long long) ncol) * sizeof(DATA_TYPE);
	DATA_TYPE *mat_gpu;
	DATA_TYPE *row_sum_vec_gpu;
	cudaMalloc((void **) &mat_gpu, mat_size);
	cudaMemcpy(mat_gpu, mat[0], mat_size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &row_sum_vec_gpu, nrow * sizeof(DATA_TYPE));
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (nrow/dimBlock.x) + (!(nrow%dimBlock.x)?0:1) );
	sum_rows<<<dimGrid,dimBlock>>>(mat_gpu, row_sum_vec_gpu, nrow, ncol);
	cudaMemcpy(row_sum_vec, row_sum_vec_gpu, nrow * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	cudaFree(mat_gpu);
	cudaFree(row_sum_vec_gpu);
}

// Taken from provided sample code
extern "C" void find_best_device() {
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
