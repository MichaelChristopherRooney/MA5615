#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include "matrix.h"

extern void do_gpu_row_sum(DATA_TYPE **mat, DATA_TYPE *row_sum_vec, int nrow, int ncol, int block_size);
extern void do_gpu_col_sum(DATA_TYPE **mat, DATA_TYPE *col_sum_vec, int nrow, int ncol, int block_size);
extern void find_best_device();

// These are the default values
static long long SEED = 123456L;
static int NROWS = 10;
static int NCOLS = 10;
static int PRINT_TIMING = 0;
static int PRINT_VALUES = 0;
int BLOCK_SIZE = 8;

void print_config(int seed_set, int nrows_set, int ncols_set, int block_size_set){
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
	if(block_size_set){
		printf("Block size set to %d\n", BLOCK_SIZE);
	} else {
		printf("Using default block size (%d)\n", BLOCK_SIZE);
	}
	if(PRINT_TIMING){
		printf("Printing timing information\n");
	} else {
		printf("Not printing timing information\n");
	}
	if(PRINT_VALUES){
		printf("Printing matrix and vector results\n");
	} else {
		printf("Not printing matrix and vector results\n");
	}
}

void parse_args(int argc, char *argv[]){
	int seed_set = 0;
	int nrows_set = 0;
	int ncols_set = 0;
	int block_size_set = 0;
	char c;
	while((c = getopt(argc, argv, "bpnmrt")) != -1){
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
		case 'p':
			PRINT_VALUES = 1;
			break;
		case 't':
			PRINT_TIMING = 1;
			break;
		case 'b':
			BLOCK_SIZE = atoi(argv[optind]);
			block_size_set = 1;
			break;
		default: // should never get here
			break;
		}
	}
	print_config(seed_set, nrows_set, ncols_set, block_size_set);
}

struct results_s {
	// CPU results
	long long cpu_row_sum_time;
	long long cpu_row_reduce_time;
	long long cpu_col_sum_time;
	long long cpu_col_reduce_time;
	DATA_TYPE *cpu_row_vec;
	DATA_TYPE cpu_col_reduce_value;
	DATA_TYPE *cpu_col_vec;
	DATA_TYPE cpu_row_reduce_value;
	// GPU results
	long long gpu_row_sum_time;
	long long gpu_col_sum_time;
	DATA_TYPE *gpu_row_vec;
	DATA_TYPE *gpu_col_vec;
};

void print_results(DATA_TYPE **mat, struct results_s *results){
	if(PRINT_TIMING){
		printf("CPU: Row summing took %lld microseconds\n", results->cpu_row_sum_time);
		printf("CPU: Column summing took %lld microseconds\n", results->cpu_col_sum_time);
		printf("CPU: Reducing row vector took %lld microseconds\n", results->cpu_row_reduce_time);
		printf("CPU: Reducing column vector took %lld microseconds\n", results->cpu_col_reduce_time);
		printf("GPU: Row summing took %lld microseconds\n", results->gpu_row_sum_time);
		printf("GPU: Col summing took %lld microseconds\n", results->gpu_col_sum_time);
	}
	if(PRINT_VALUES){
		printf("Matrix is:\n");
		print_matrix(mat, NROWS, NCOLS);
		printf("CPU: Row sum vector: ");
		print_vector(results->cpu_row_vec, NROWS);
		printf("CPU: Row sum vector reduced: %f\n", results->cpu_row_reduce_value);
		printf("CPU: Column sum vector: ");
		print_vector(results->cpu_col_vec, NCOLS);
		printf("CPU: Column sum vector reduced: %f\n", results->cpu_col_reduce_value);
		printf("GPU: Row sum vector: ");
		print_vector(results->gpu_row_vec, NROWS);
		printf("GPU: Col sum vector: ");
		print_vector(results->gpu_col_vec, NCOLS);
	}
//#define DISABLE_CPU
#ifndef DISABLE_CPU
	int diff = compare_vectors(results->cpu_row_vec, results->gpu_row_vec, NROWS, 0.000001);
	if(diff == 1){
		printf("ERROR: CPU and GPU row vectors do not match\n");
	} else {
		printf("SUCCESS: CPU and GPU row vectors match\n");
	}
	diff = compare_vectors(results->cpu_col_vec, results->gpu_col_vec, NCOLS, 0.000001);
	if(diff == 1){
		printf("ERROR: CPU and GPU col vectors do not match\n");
	} else {
		printf("SUCCESS: CPU and GPU col vectors match\n");
	}
#endif
}

void time_work(DATA_TYPE **mat){
	struct results_s *results = calloc(1, sizeof(struct results_s));
	struct timeval start, end;
#ifndef DISABLE_CPU
	// Time row summing
	gettimeofday(&start, NULL);
	results->cpu_row_vec = sum_rows_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	results->cpu_row_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Time column summing
	gettimeofday(&start, NULL);
	results->cpu_col_vec = sum_cols_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	results->cpu_col_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Time reducing the row vector
	gettimeofday(&start, NULL);
	results->cpu_row_reduce_value = reduce_vector(results->cpu_row_vec, NROWS);
	gettimeofday(&end, NULL);
	results->cpu_row_reduce_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Time reducing the column vector
	gettimeofday(&start, NULL);
	results->cpu_col_reduce_value = reduce_vector(results->cpu_col_vec, NCOLS);
	gettimeofday(&end, NULL);
	results->cpu_col_reduce_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
#endif
	// Time GPU row summing
	results->gpu_row_vec = (DATA_TYPE *) malloc(NROWS * sizeof(DATA_TYPE));
	gettimeofday(&start, NULL);
	do_gpu_row_sum(mat, results->gpu_row_vec, NROWS, NCOLS, BLOCK_SIZE);
	gettimeofday(&end, NULL);
	results->gpu_row_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Time GPU col summing
	results->gpu_col_vec = (DATA_TYPE *) malloc(NCOLS * sizeof(DATA_TYPE));
	gettimeofday(&start, NULL);
	do_gpu_col_sum(mat, results->gpu_col_vec, NROWS, NCOLS, BLOCK_SIZE);
	gettimeofday(&end, NULL);
	results->gpu_col_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Now print results and cleanup
	print_results(mat, results);
	free(results->cpu_row_vec);
	free(results->cpu_col_vec);
	free(results);
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	find_best_device();
	srand48(SEED);
	DATA_TYPE **mat = create_random_matrix(NROWS, NCOLS);
	time_work(mat);
	DATA_TYPE *row_sum_vec = (DATA_TYPE *) malloc(NROWS * sizeof(DATA_TYPE));
	DATA_TYPE *col_sum_vec = (DATA_TYPE *) malloc(NCOLS * sizeof(DATA_TYPE));
	/*do_gpu_sums(mat, row_sum_vec, col_sum_vec, NROWS, NCOLS);
	printf("GPU: Row sum vector: ");
	print_vector(row_sum_vec, NROWS);
	printf("GPU: Column sum vector: ");
	print_vector(col_sum_vec, NCOLS);*/
	free_matrix(mat);
	return 0;
}

