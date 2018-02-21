#include "stdlib.h"
#include "stdio.h"

#include "matrix.h"

__global__ void sum_rows(float *mat, float *out, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nrow){
		return;
	}
	float result = 0.0;
	for(int i = 0; i < ncol; i++){
		int index = (ncol * idx) + i;
		//printf("Thread %d accessing %d\n", idx, index);
		result += mat[index];
	}
	out[idx] = result;
	//printf("Thread %d got %f\n", idx, result);
}

__global__ void sum_columns(float *mat, float *out, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= ncol){
		return;
	}
	float result = 0.0;
	for(int i = 0; i < nrow; i++){
		int index = (ncol * i) + idx;
		//printf("Thread %d accessing %d\n", idx, index);
		result += mat[index];
	}
	out[idx] = result;
	//printf("Thread %d got %f\n", idx, result);
}

extern "C" void do_gpu_col_sum(float **mat, float *col_sum_vec, int nrow, int ncol){
	int mat_size = sizeof(float) * nrow * ncol;
	float *mat_gpu;
	float *col_sum_vec_gpu;
	cudaMalloc((void **) &mat_gpu, mat_size);
	cudaMalloc((void **) &col_sum_vec_gpu, ncol * sizeof(float));
	cudaMemcpy(mat_gpu, mat[0], mat_size, cudaMemcpyHostToDevice);
	int block_size = 32;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (mat_size/dimBlock.x) + (!(ncol%dimBlock.x)?0:1) );
	sum_columns<<<dimGrid,dimBlock>>>(mat_gpu, col_sum_vec_gpu, nrow, ncol);
	cudaMemcpy(col_sum_vec, col_sum_vec_gpu, ncol * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(mat_gpu);
	cudaFree(col_sum_vec);
}

extern "C" void do_gpu_row_sum(float **mat, float *row_sum_vec, int nrow, int ncol){
	int mat_size = sizeof(float) * nrow * ncol;
	float *mat_gpu;
	float *row_sum_vec_gpu;
	cudaMalloc((void **) &mat_gpu, mat_size);
	cudaMalloc((void **) &row_sum_vec_gpu, nrow * sizeof(float));
	cudaMemcpy(mat_gpu, mat[0], mat_size, cudaMemcpyHostToDevice);
	int block_size = 32;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (mat_size/dimBlock.x) + (!(nrow%dimBlock.x)?0:1) );
	sum_rows<<<dimGrid,dimBlock>>>(mat_gpu, row_sum_vec_gpu, nrow, ncol);
	cudaMemcpy(row_sum_vec, row_sum_vec_gpu, nrow * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(mat_gpu);
	cudaFree(row_sum_vec);
}

