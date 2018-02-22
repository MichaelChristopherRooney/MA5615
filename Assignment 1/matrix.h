int compare_vectors(DATA_TYPE *vec1, DATA_TYPE *vec2, int len, DATA_TYPE epsilon);
void print_matrix(DATA_TYPE **mat, int nrow, int ncol);
void free_matrix(DATA_TYPE **mat);

DATA_TYPE **create_empty_matrix(int nrow, int ncol);
DATA_TYPE **create_random_matrix(int nrow, int ncol);
void sum_rows_to_vector(DATA_TYPE **mat, DATA_TYPE *vec, DATA_TYPE *reduced, int nrow, int ncol);
void sum_cols_to_vector(DATA_TYPE **mat, DATA_TYPE *vec, DATA_TYPE *reduced, int nrow, int ncol);
void print_vector(DATA_TYPE *vec, int len);
DATA_TYPE reduce_vector(DATA_TYPE *vec, int len);
