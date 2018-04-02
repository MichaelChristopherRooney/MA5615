#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "grid.h"

void print_grid(DATA_TYPE **grid, int nrow, int ncol){
	int i, n;
	for(i = 0; i < nrow; i++){
		for(n = 0; n < ncol; n++){
			printf("%f, ", grid[i][n]);
		}
		printf("\n");
	}
}

void print_reduce(DATA_TYPE *reduce, int nrow){
	int i;
	for(i = 0; i < nrow; i++){
		printf("%f, ", reduce[i]);
	}
	printf("\n");
}

void free_grid(DATA_TYPE **grid){
	free(grid[0]);
	free(grid);
}

DATA_TYPE **create_empty_grid(int nrow, int ncol){
	DATA_TYPE **grid = (DATA_TYPE **) malloc(sizeof(DATA_TYPE *) * nrow);
	DATA_TYPE *temp = (DATA_TYPE *) calloc(sizeof(DATA_TYPE), nrow * ncol);
	int i;
	for (i = 0; i < nrow; i++) {
		grid[i] = &(temp[ncol * i]);
	}
	return grid;
}

DATA_TYPE **init_grid(int nrow, int ncol){
	DATA_TYPE **grid = create_empty_grid(nrow, ncol);
	int i;
	for(i = 0; i < nrow; i++){
		grid[i][0] = 0.85*(DATA_TYPE)((i+1)*(i+1)) / (DATA_TYPE)(nrow * nrow);
		grid[i][1] = (DATA_TYPE)((i+1)*(i+1)) / (DATA_TYPE)(nrow * nrow);
	}
	return grid;
}

// Returns 0 if the grids match, otherwise 1
int compare_grids(DATA_TYPE **g1, DATA_TYPE **g2, int nrow, int ncol){
	static DATA_TYPE epsilon = 1.E-5;
	int i, j;
	for(i = 0; i < nrow; i++){
		for(j = 0; j < ncol; j++){
			DATA_TYPE d1 = g1[i][j];
			DATA_TYPE d2 = g2[i][j];
			if(fabs(d1 - d2) > epsilon){
				printf("ERROR: CPU grid has %f, but CUDA grid has %f\n", d1, d2);
				return 1;
			}
		}
	}
	return 0;
}

// Returns 0 if the reductions match, otherwise 1
int compare_reductions(DATA_TYPE *r1, DATA_TYPE *r2, int nrow){
	static DATA_TYPE epsilon = 1.E-5;
	int i;
	for(i = 0; i < nrow; i++){
		DATA_TYPE d1 = r1[i];
		DATA_TYPE d2 = r2[i];
		if(fabs(d1 - d2) > epsilon){
			printf("ERROR: CPU reduction has %f, but CUDA reduction has %f\n", d1, d2);
			return 1;
		}
	}
	return 0;
}
