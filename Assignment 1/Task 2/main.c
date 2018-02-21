#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include "matrix.h"

extern void do_gpu_sums(float **mat, float *row_sum_vec, float *col_sum_vec, int nrow, int ncol);

// These are the default values
static long long SEED = 123456L;
static int NROWS = 10;
static int NCOLS = 10;
static int PRINT_TIMING = 0;

void print_config(int seed_set, int nrows_set, int ncols_set){
	if(seed_set){
		printf("Set seed to current time in microseconds (%lld)\n", SEED);
	} else {
		printf("Using default seed (%lld)\n", SEED);
	}
	if(nrows_set){
		printf("Nrows set to %d\n", NROWS);
	} else {
		printf("Using default nrows (%d)\n", NROWS);
	}
	if(ncols_set){
		printf("Ncols set to %d\n", NCOLS);
	} else {
		printf("Using default ncols (%d)\n", NCOLS);
	}
	if(PRINT_TIMING){
		printf("Printing timing information\n");
	} else {
		printf("Not printing timing information\n");
	}
}

void parse_args(int argc, char *argv[]){
	int seed_set = 0;
	int nrows_set = 0;
	int ncols_set = 0;
	char c;
	while((c = getopt(argc, argv, "nmrt")) != -1){
		if(c == '?'){
			printf("ERROR: unknown flag passed\n");
			exit(1);
		}
		switch(c){
		case 'n':
			NROWS = atoi(argv[optind]);
			nrows_set = 1;
			break;
		case 'm':
			NCOLS = atoi(argv[optind]);
			ncols_set = 1;
			break;
		case 'r': {
			struct timeval tv;
			gettimeofday(&tv, NULL);
			SEED = tv.tv_usec;
			seed_set = 1;
			break;
		}
		case 't':
			PRINT_TIMING = 1;
			break;
		default: // should never get here
			break;
		}
	}
	print_config(seed_set, nrows_set, ncols_set);
}

void do_sums_cpu(float **mat){
	struct timeval start, end;
	// Time row summing
	gettimeofday(&start, NULL);
	float *row_vec = sum_rows_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	long long time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("CPU: Row summing took %lld microseconds\n", time_taken);
	}
	// Now time column summing
	gettimeofday(&start, NULL);
	float *col_vec = sum_cols_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("CPU: Column summing took %lld microseconds\n", time_taken);
	}
	// Time reducing the row vector
	gettimeofday(&start, NULL);
	float row_sum = reduce_vector(row_vec, NROWS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("CPU: Reducing row vector took %lld microseconds\n", time_taken);
	}
	// Time reducing the column vector
	gettimeofday(&start, NULL);
	float col_sum = reduce_vector(col_vec, NCOLS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("CPU: Reducing column vector took %lld microseconds\n", time_taken);
	}
	printf("CPU: Row sum vector: ");
	print_vector(row_vec, NROWS);
	printf("CPU: Row sum vector reduced: %f\n", row_sum);
	printf("CPU: Column sum vector: ");
	print_vector(col_vec, NCOLS);
	printf("CPU: Column sum vector reduced: %f\n", col_sum);
	free(row_vec);
	free(col_vec);
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	srand48(SEED);
	float **mat = create_random_matrix(NROWS, NCOLS);
	print_matrix(mat, NROWS, NCOLS);
	do_sums_cpu(mat);
	float *row_sum_vec = (float *) malloc(NROWS * sizeof(float));
	float *col_sum_vec = (float *) malloc(NCOLS * sizeof(float));
	do_gpu_sums(mat, row_sum_vec, col_sum_vec, NROWS, NCOLS);
	printf("GPU: Row sum vector: ");
	print_vector(row_sum_vec, NROWS);
	printf("GPU: Column sum vector: ");
	print_vector(col_sum_vec, NCOLS);
	free_matrix(mat);
	return 0;
}
