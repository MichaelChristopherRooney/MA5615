#include "data_type.h"

void print_grid(DATA_TYPE **grid, int nrow, int ncol);
void print_reduce(DATA_TYPE *reduce, int nrow);
void free_grid(DATA_TYPE **grid);
DATA_TYPE **create_empty_grid(int nrow, int ncol);
DATA_TYPE **init_grid(int nrow, int ncol);
int compare_grids(DATA_TYPE **g1, DATA_TYPE **g2, int nrow, int ncol);
int compare_reductions(DATA_TYPE *r1, DATA_TYPE *r2, int nrow);
