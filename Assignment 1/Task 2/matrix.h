void print_matrix(float **mat, int nrow, int ncol);
void free_matrix(float **mat);
float **create_empty_matrix(int nrow, int ncol);
float **create_random_matrix(int nrow, int ncol);
float *sum_rows_to_vector(float **mat, int nrow, int ncol);
float *sum_cols_to_vector(float **mat, int nrow, int ncol);
void print_vector(float *vec, int len);
float reduce_vector(float *vec, int len);
