#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include "matrix.h"

extern void do_gpu_row_sum(float **mat, float *row_sum_vec, int nrow, int ncol);
extern void do_gpu_col_sum(float **mat, float *col_sum_vec, int nrow, int ncol);

// These are the default values
static long long SEED = 123456L;
static int NROWS = 10;
static int NCOLS = 10;
static int PRINT_TIMING = 0;
static int PRINT_VALUES = 0;

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
	char c;
	while((c = getopt(argc, argv, "pnmrt")) != -1){
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
		default: // should never get here
			break;
		}
	}
	print_config(seed_set, nrows_set, ncols_set);
}

struct results_s {
	// CPU results
	long long cpu_row_sum_time;
	long long cpu_row_reduce_time;
	long long cpu_col_sum_time;
	long long cpu_col_reduce_time;
	float *cpu_row_vec;
	float cpu_col_reduce_value;
	float *cpu_col_vec;
	float cpu_row_reduce_value;
	// GPU results
	long long gpu_row_sum_time;
	long long gpu_col_sum_time;
	float *gpu_row_vec;
	float *gpu_col_vec;
};

void print_results(float **mat, struct results_s *results){
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
		print_vector(results->gpu_col_vec, NROWS);
	}
	int diff = compare_vectors(results->cpu_row_vec, results->gpu_row_vec, NROWS, 0.000001);
	if(diff == 1){
		printf("ERROR: CPU and GPU row vectors do not match\n");
	} else {
		printf("SUCCESS: CPU and GPU row vectors match\n");
	}
	diff = compare_vectors(results->cpu_col_vec, results->gpu_col_vec, NROWS, 0.000001);
	if(diff == 1){
		printf("ERROR: CPU and GPU col vectors do not match\n");
	} else {
		printf("SUCCESS: CPU and GPU col vectors match\n");
	}
}

void time_work(float **mat){
	struct results_s *results = calloc(1, sizeof(struct results_s));
	struct timeval start, end;
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
	// Time GPU row summing
	results->gpu_row_vec = (float *) malloc(NROWS * sizeof(float));
	gettimeofday(&start, NULL);
	do_gpu_row_sum(mat, results->gpu_row_vec, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	results->gpu_row_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	// Time GPU col summing
	results->gpu_col_vec = (float *) malloc(NCOLS * sizeof(float));
	printf("GPU col vec pointer: %p\n", results->gpu_col_vec);
	gettimeofday(&start, NULL);
	do_gpu_col_sum(mat, results->gpu_col_vec, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	results->gpu_col_sum_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	/*
	long long gpu_row_sum_time;
	long long gpu_col_sum_time;
	float *gpu_row_vec;
	float *gpu_col_vec;
	*/
	// Now print results and cleanup
	print_results(mat, results);
	free(results->cpu_row_vec);
	free(results->cpu_col_vec);
	free(results);
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	srand48(SEED);
	float **mat = create_random_matrix(NROWS, NCOLS);
	time_work(mat);
	float *row_sum_vec = (float *) malloc(NROWS * sizeof(float));
	float *col_sum_vec = (float *) malloc(NCOLS * sizeof(float));
	/*do_gpu_sums(mat, row_sum_vec, col_sum_vec, NROWS, NCOLS);
	printf("GPU: Row sum vector: ");
	print_vector(row_sum_vec, NROWS);
	printf("GPU: Column sum vector: ");
	print_vector(col_sum_vec, NCOLS);*/
	free_matrix(mat);
	return 0;
}

