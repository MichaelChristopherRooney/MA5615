#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"

int compare_vectors(DATA_TYPE *vec1, DATA_TYPE *vec2, int len, DATA_TYPE epsilon){
	int i;
	for(i = 0; i < len; i++){
		DATA_TYPE diff = fabs(vec1[i] - vec2[i]);
		if(diff > epsilon){
			return 1;
		}
	}
	return 0;
}

void print_matrix(DATA_TYPE **mat, int nrow, int ncol){
	int i, n;
	for(i = 0; i < nrow; i++){
		for(n = 0; n < ncol; n++){
			printf("%f, ", mat[i][n]);
		}
		printf("\n");
	}
}

void free_matrix(DATA_TYPE **mat){
	free(mat[0]);
	free(mat);
}

DATA_TYPE **create_empty_matrix(int nrow, int ncol){
	DATA_TYPE **mat = (DATA_TYPE **) malloc(sizeof(DATA_TYPE *) * nrow);
	DATA_TYPE *temp = (DATA_TYPE *) calloc(sizeof(DATA_TYPE), nrow * ncol);
	int i;
	for (i = 0; i < nrow; i++) {
		mat[i] = &(temp[ncol * i]);
	}
	return mat;
}

DATA_TYPE **create_random_matrix(int nrow, int ncol){
	DATA_TYPE **mat = create_empty_matrix(nrow, ncol);
	int i, n;
	for(i = 0; i < nrow; i++){
		for(n = 0; n < ncol; n++){
			mat[i][n] = (DATA_TYPE) drand48();
		}
	}
	return mat;
}

DATA_TYPE reduce_vector(DATA_TYPE *vec, int len){
	DATA_TYPE sum = 0.0;
	int i;
	for(i = 0; i < len; i++){
		sum += vec[i];
	}
	return sum;
}

void print_vector(DATA_TYPE *vec, int len){
	int i;
	for(i = 0; i < len; i++){
		printf("%f, ", vec[i]);
	}
	printf("\n");
}

DATA_TYPE *sum_rows_to_vector(DATA_TYPE **mat, int nrow, int ncol){
	DATA_TYPE *vec = (DATA_TYPE *) calloc(sizeof(DATA_TYPE), nrow);
	int i, n;
	for(i = 0; i < nrow; i++){
		DATA_TYPE row_sum = 0.0;
		for(n = 0; n < ncol; n++){
			row_sum += mat[i][n];
		}
		vec[i] = row_sum;
	}
	//print_vector(vec, nrow);
	return vec;
}

DATA_TYPE *sum_cols_to_vector(DATA_TYPE **mat, int nrow, int ncol){
	DATA_TYPE *vec = (DATA_TYPE *) calloc(sizeof(DATA_TYPE), ncol);
	int i, n;
	for(i = 0; i < ncol; i++){
		DATA_TYPE col_sum = 0.0;
		for(n = 0; n < nrow; n++){
			col_sum += mat[n][i];
		}
		vec[i] = col_sum;
	}
	//print_vector(vec, ncol);
	return vec;
}

