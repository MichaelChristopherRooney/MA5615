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
