#include "stdlib.h"
#include "stdio.h"
#include <string>
#include <iostream>

#include "grid.h"

void custom_error_check(cudaError result, std::string err_str){
	if(result != cudaSuccess){
		std::cout << err_str << "\n";
		std::cout << "Error code: " << result << "\n";
		exit(1);
	}	
}

// Each thread takes one row
// Uses mostly global memory
__global__ void cuda_do_grid_iterations_naive_ver(DATA_TYPE *grid_gpu_1, DATA_TYPE *grid_gpu_2, int nrow, int ncol, int num_iter){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
        long long row_offset = idx * ncol;
        DATA_TYPE *cur = grid_gpu_1;
        DATA_TYPE *next = grid_gpu_2;
        if(idx >= nrow){
                return;
        }
        for(int i = 0; i < num_iter; i++){
                for(long long n = 2; n < ncol; n++){
                        next[row_offset+n] = 0.15*(cur[row_offset+n-2]);
                        next[row_offset+n] += 0.65*(cur[row_offset+n-1]);
                        next[row_offset+n] += (cur[row_offset+n]);
                        if(n == ncol - 2){
                                next[row_offset+n] += 1.35*(cur[row_offset+n+1]);
                                next[row_offset+n] += 1.85*(cur[row_offset]);
                        } else if(n == ncol - 1){
                                next[row_offset+n] += 1.35*(cur[row_offset]);
                                next[row_offset+n] += 1.85*(cur[row_offset+1]);
                        } else {
                                next[row_offset+n] += 1.35*(cur[row_offset+n+1]);
                                next[row_offset+n] += 1.85*(cur[row_offset+n+2]);
                        }
                        next[row_offset+n] = next[row_offset+n] / 5.0;
                }
                DATA_TYPE *temp = cur;
                cur = next;
                next = temp;
        }
	if(cur != grid_gpu_1){
		for(int n = 2; n < ncol; n++){
			grid_gpu_1[row_offset+n] = grid_gpu_2[row_offset+n];
		}
	}
}

extern "C" DATA_TYPE *do_grid_iterations_gpu_naive_ver(DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size, int num_iter){
	long long grid_size = (long long) nrow * (long long) ncol * (long long) sizeof(DATA_TYPE);
	DATA_TYPE *grid_gpu_device_1;
	DATA_TYPE *grid_gpu_device_2;
	custom_error_check(cudaMalloc((void **) &grid_gpu_device_1, grid_size), "Failed to allocate on device");
	custom_error_check(cudaMalloc((void **) &grid_gpu_device_2, grid_size), "Failed to allocate on device");
	custom_error_check(cudaMemcpy(grid_gpu_device_1, grid_gpu_host[0], grid_size, cudaMemcpyHostToDevice), "Failed to copy data to device");
	custom_error_check(cudaMemcpy(grid_gpu_device_2, grid_gpu_host[0], grid_size, cudaMemcpyHostToDevice), "Failed to copy data to device");
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (nrow/dimBlock.x) + (!(nrow%dimBlock.x)?0:1) );
	cuda_do_grid_iterations_naive_ver<<<dimGrid,dimBlock>>>(grid_gpu_device_1, grid_gpu_device_2, nrow, ncol, num_iter);
	custom_error_check(cudaPeekAtLastError(), "Error during kernel execution");
	custom_error_check(cudaMemcpy(grid_gpu_host[0], grid_gpu_device_1, grid_size, cudaMemcpyDeviceToHost), "Failed to copy data FROM device");
	//custom_error_check(cudaFree(grid_gpu_device_1), "Failed to free memory on device");
	custom_error_check(cudaFree(grid_gpu_device_2), "Failed to free memory on device");
	return grid_gpu_device_1;
}

// See report for details on the optimisations used here.
// In a nutshell this version:
// 1) greatly reduces the number of memory operations by using registers
// 2) each row calculates its own constant values, meaning the grid does not need to copied TO the device
//	^ note the the grid must still be copied FROM the device
// 3) replace grid[i] /= 5.0 with grid[i] *= (1.0/5.0), where 1.0/5.0 is calculated once at the start.
// 4) unroll inner loop so two columns are processed per inner loop iteration
// 5) uses float2/double2 to store results that are written to memory, reducing number of memory ops
__global__ void cuda_do_grid_iterations_fast_ver(DATA_TYPE *grid_gpu, int nrow, int ncol, int num_iter){
	long long idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= (nrow)){
		return;
	}
	const DATA_TYPE one_over_five = 1.0 / 5.0;
	const long long offset = idx * (long long) ncol;
	// Constants for the first row (row 0)
	const DATA_TYPE col_0_fixed_val = 0.85*(DATA_TYPE)((idx+1)*(idx+1)) / (DATA_TYPE)(nrow * nrow);
	const DATA_TYPE col_1_fixed_val= (DATA_TYPE)((idx+1)*(idx+1)) / (DATA_TYPE)(nrow * nrow);
	// Place the constants in the first two columns of the grid.
	grid_gpu[offset] = col_0_fixed_val;
	grid_gpu[offset+1] = col_1_fixed_val;
	float2 result;
	float2 next_vals;
	for(int i = 0; i < num_iter; i++){
		// Registers for the first row (row 0)
		DATA_TYPE val_n_minus_2 = col_0_fixed_val;
		DATA_TYPE val_n_minus_1 = col_1_fixed_val;
		DATA_TYPE val = grid_gpu[offset + 2];
		DATA_TYPE val_orig = val;
		DATA_TYPE val_n_plus_1 = grid_gpu[offset + 3];
		DATA_TYPE val_n_plus_2 = grid_gpu[offset + 4];
		DATA_TYPE val_next;
		for(int n = 2; n < ncol - 4; n = n + 2){
			next_vals = *(float2*)(&grid_gpu[offset+n+3]);
			// Unrolled part 0
			val += 0.15*val_n_minus_2;
			val += 0.65*val_n_minus_1;
			val_next = val_n_plus_1;
			val += 1.35*val_n_plus_1;
			val += 1.85*val_n_plus_2;
			val_n_plus_1 = val_n_plus_2;
			val_n_plus_2 = next_vals.x;
			val = val * one_over_five;
			result.x = val;
			val_n_minus_2 = val_n_minus_1;
			val_n_minus_1 = val_orig;
			val = val_next;
			val_orig = val_next;
			// Unrolled part 1
			val += 0.15*val_n_minus_2;
			val += 0.65*val_n_minus_1;
			val_next = val_n_plus_1;
			val += 1.35*val_n_plus_1;
			val += 1.85*val_n_plus_2;
			val_n_plus_1 = val_n_plus_2;
			val_n_plus_2 = next_vals.y;
			val = val * one_over_five;
			result.y = val;
			val_n_minus_2 = val_n_minus_1;
			val_n_minus_1 = val_orig;
			val = val_next;
			val_orig = val_next;
			*(float2*)(&grid_gpu[offset+n]) = result;
		}
		next_vals = *(float2*)(&grid_gpu[offset+ncol-2]);
		// Fourth last column
		val += 0.15*val_n_minus_2;
		val += 0.65*val_n_minus_1;
		val_next = val_n_plus_1;
		val += 1.35*val_n_plus_1;
		val += 1.85*val_n_plus_2;
		val_n_plus_1 = val_n_plus_2;
		val_n_plus_2 = next_vals.y;
		val = val * one_over_five;
		result.x = val;
		val_n_minus_2 = val_n_minus_1;
		val_n_minus_1 = val_orig;
		val = val_next;
		val_orig = val_next;
		// Third last column
		val += 0.15*val_n_minus_2;
		val += 0.65*val_n_minus_1;
		val_next = val_n_plus_1;
		val += 1.35*val_n_plus_1;
		val += 1.85*val_n_plus_2;
		val_n_plus_1 = val_n_plus_2;
		val = val * one_over_five;
		result.y = val;
		val_n_minus_2 = val_n_minus_1;
		val_n_minus_1 = val_orig;
		val = val_next;
		val_orig = val_next;
		*(float2*)(&grid_gpu[offset+ncol-4]) = result;
		// Second last column
		val += 0.15*val_n_minus_2;
		val += 0.65*val_n_minus_1;
		val_next = val_n_plus_1;
		val += 1.35*val_n_plus_1;
		val += 1.85*col_0_fixed_val;
		val = val * one_over_five;
		result.x = val;
		val_n_minus_2 = val_n_minus_1;
		val_n_minus_1 = val_orig;
		val = val_next;
		// Last column
		val += 0.15*val_n_minus_2;
		val += 0.65*val_n_minus_1;
		val_next = val_n_plus_1;
		val += 1.35*col_0_fixed_val;
		val += 1.85*col_1_fixed_val;
		val = val * one_over_five;
		result.y = val;
		*(float2*)(&grid_gpu[offset+ncol-2]) = result;
	}
}

extern "C" DATA_TYPE* do_grid_iterations_gpu_fast_ver(DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size, int num_iter){
	long long grid_size = (long long) (nrow) * (long long) (ncol) * (long long) sizeof(DATA_TYPE);
	DATA_TYPE *grid_gpu_device;
	custom_error_check(cudaMalloc((void **) &grid_gpu_device, grid_size), "Failed to allocate on device");
	dim3 dimBlock(block_size);
	dim3 dimGrid ( ((nrow)/dimBlock.x) + (!((nrow)%dimBlock.x)?0:1) );
	cuda_do_grid_iterations_fast_ver<<<dimGrid,dimBlock>>>(grid_gpu_device, nrow, ncol, num_iter);
	custom_error_check(cudaPeekAtLastError(), "Error during kernel execution");
	custom_error_check(cudaMemcpy(grid_gpu_host[0], grid_gpu_device, grid_size, cudaMemcpyDeviceToHost), "Failed to copy data FROM device");
	//custom_error_check(cudaFree(grid_gpu_device), "Failed to free memory on device");
	return grid_gpu_device;
}

__global__ void cuda_do_reduce_naive(DATA_TYPE *grid_device, DATA_TYPE *reduce_device, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nrow){
		return;
	}
	const int offset = idx * ncol;
	for(int i = 0; i < ncol; i++){
		reduce_device[idx] += grid_device[offset+i];
	}
}

extern "C" void do_reduce_naive(DATA_TYPE *grid_device, DATA_TYPE *reduce_host, int nrow, int ncol, int block_size){
	long long reduce_size = (long long) nrow * (long long) sizeof(DATA_TYPE);
	DATA_TYPE *reduce_device;
	custom_error_check(cudaMalloc((void **) &reduce_device, reduce_size), "Failed to allocate on device");
	dim3 dimBlock(block_size);
	dim3 dimGrid ( ((nrow)/dimBlock.x) + (!((nrow)%dimBlock.x)?0:1) );
	cuda_do_reduce_naive<<<dimGrid,dimBlock>>>(grid_device, reduce_device, nrow, ncol);
	custom_error_check(cudaPeekAtLastError(), "Error during kernel execution");
	custom_error_check(cudaMemcpy(reduce_host, reduce_device, reduce_size, cudaMemcpyDeviceToHost), "Failed to copy data FROM device");
	custom_error_check(cudaFree(reduce_device), "Failed to free memory on device");
}

__global__ void cuda_do_reduce_fast(DATA_TYPE *grid_device, DATA_TYPE *reduce_device, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nrow){
		return;
	}
	const int offset = idx * ncol;
	DATA_TYPE val;
	DATA_TYPE sum = 0.0;
	float4 vals;
	for(int i = 0; i < ncol; i += 4){
		vals = *(float4*)(&grid_device[offset+i]);
		val = vals.x + vals.y + vals.z + vals.w;
		sum += val;
	}
	reduce_device[idx] = sum;
}

extern "C" void do_reduce_fast(DATA_TYPE *grid_device, DATA_TYPE *reduce_host, int nrow, int ncol, int block_size){
	long long reduce_size = (long long) nrow * (long long) sizeof(DATA_TYPE);
	DATA_TYPE *reduce_device;
	custom_error_check(cudaMalloc((void **) &reduce_device, reduce_size), "Failed to allocate on device");
	dim3 dimBlock(block_size);
	dim3 dimGrid ( ((nrow)/dimBlock.x) + (!((nrow)%dimBlock.x)?0:1) );
	cuda_do_reduce_fast<<<dimGrid,dimBlock>>>(grid_device, reduce_device, nrow, ncol);
	custom_error_check(cudaPeekAtLastError(), "Error during kernel execution");
	custom_error_check(cudaMemcpy(reduce_host, reduce_device, reduce_size, cudaMemcpyDeviceToHost), "Failed to copy data FROM device");
	custom_error_check(cudaFree(reduce_device), "Failed to free memory on device");
}

extern "C" void free_on_device(DATA_TYPE *device_ptr){
	custom_error_check(cudaFree(device_ptr), "Failed to free memory on device");
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
