#include "stdlib.h"
#include "stdio.h"

#include "matrix.h"

__global__ void sum_columns(float *mat, int nrow, int ncol){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= ncol){
		return;
	}
	float result = 0.0;
	for(int i = 0; i < nrow; i++){
		int index = (nrow * i) + idx;
		printf("Thread %d accessing %d\n", idx, index);
		result += mat[index];
	}
	printf("Thread %d got %f\n", idx, result);
}

// TODO: seed and other args
int main(int argc, char *argv[]){
	srand48(123456);
	int ncol = 5;
	int nrow = 5;
	int mat_size = sizeof(float) * nrow * ncol;
	float **mat = create_random_matrix(nrow, ncol);
	float *mat_gpu;
	cudaMalloc((void **) &mat_gpu, mat_size);
	cudaMemcpy(mat_gpu, mat[0], mat_size, cudaMemcpyHostToDevice);
	printf("On host matrix is:\n");
	print_matrix(mat, nrow, ncol);
	int block_size = 8;
	dim3 dimBlock(block_size);
	dim3 dimGrid ( (mat_size/dimBlock.x) + (!(mat_size%dimBlock.x)?0:1) );
	sum_columns<<<dimGrid,dimBlock>>>(mat_gpu, nrow, ncol);
	free_matrix(mat);
	cudaFree(mat_gpu);
	return 0;
}
