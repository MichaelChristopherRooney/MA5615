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
		grid[i][0] = 0.85*(float)((i+1)*(i+1)) / (float)(nrow * nrow);
		grid[i][1] = (float)((i+1)*(i+1)) / (float)(nrow * nrow);
	}
	return grid;
}





