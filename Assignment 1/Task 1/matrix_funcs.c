#include <stdlib.h>
#include <stdio.h>

#include "matrix_funcs.h"

void print_matrix(float **mat, int nrow, int ncol){
	int i, n;
	for(i = 0; i < nrow; i++){
		for(n = 0; n < ncol; n++){
			printf("%f, ", mat[i][n]);
		}
		printf("\n");
	}
}

void free_matrix(float **mat){
	free(mat[0]);
	free(mat);
}	

float **create_empty_matrix(int nrow, int ncol){
	float **mat = malloc(sizeof(float *) * nrow);
	float *temp = calloc(sizeof(float), nrow * ncol);
	int i;
	for (i = 0; i < nrow; i++) {
		mat[i] = &(temp[ncol * i]);
	}
	return mat;
}

float **create_random_matrix(int nrow, int ncol){
	float **mat = create_empty_matrix(nrow, ncol);
	int i, n;
	for(i = 0; i < nrow; i++){
		for(n = 0; n < ncol; n++){
			mat[i][n] = (float) drand48();
		}
	}
	return mat;
}

float reduce_vector(float *vec, int len){
	float sum = 0.0;
	int i;
	for(i = 0; i < len; i++){
		sum += vec[i];
	}
	return sum;
}

void print_vector(float *vec, int len){
	int i;
	for(i = 0; i < len; i++){
		printf("%f, ", vec[i]);
	}
	printf("\n");
}

float *sum_rows_to_vector(float **mat, int nrow, int ncol){
	float *vec = calloc(sizeof(float), nrow);
	int i, n;
	for(i = 0; i < nrow; i++){
		float row_sum = 0.0;
		for(n = 0; n < ncol; n++){
			row_sum += mat[i][n];
		}
		vec[i] = row_sum;
	}
	//print_vector(vec, nrow);
	return vec;
}

float *sum_cols_to_vector(float **mat, int nrow, int ncol){
	float *vec = calloc(sizeof(float), ncol);
	int i, n;
	for(i = 0; i < ncol; i++){
		float col_sum = 0.0;
		for(n = 0; n < nrow; n++){
			col_sum += mat[n][i];
		}
		vec[i] = col_sum;
	}
	//print_vector(vec, ncol);
	return vec;
}
