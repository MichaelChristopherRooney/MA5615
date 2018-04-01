#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "grid.h"

// CUDA functions from gpu_code.cu
extern void find_best_device();
extern void do_grid_iterations_gpu_naive_ver(DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size, int num_iter);
extern void do_grid_iterations_gpu_fast_ver(DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size, int num_iter);
extern void do_reduce_naive(DATA_TYPE **grid_gpu_host, DATA_TYPE *reduce_host, int nrow, int ncol, int block_size);
extern void do_reduce_fast(DATA_TYPE **grid_gpu_host, DATA_TYPE *reduce_host, int nrow, int ncol, int block_size);

// These are the default values
static int NROWS = 32;
static int NCOLS = 32;
static int NUM_ITERATIONS = 10;
static int PRINT_TIMING = 0;
static int PRINT_VALUES = 0;
static int AVERAGE_ROWS = 0;
static int SKIP_CPU = 0;
static int SKIP_CUDA = 0;
int BLOCK_SIZE = 64;

void print_config(int nrows_set, int ncols_set, int block_size_set, int iter_set){
	printf("==========\nCONFIGURATION\n==========\n");
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
	if(iter_set){
		printf("Number of iterations set to %d\n", NUM_ITERATIONS);
	} else {
		printf("Using default number of iterations (%d)\n", NUM_ITERATIONS);
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
		printf("Printing grid results\n");
	} else {
		printf("Not printing grid results\n");
	}
	if(AVERAGE_ROWS){
		printf("Averaging row temperature\n");
	} else {
		printf("Not averaging row temperature\n");
	}
	if(SKIP_CPU){
		printf("Not executing CPU version\n");
	}
	if(SKIP_CUDA){
		printf("Not executing CUDA version\n");
	}
	printf("==========\nEND CONFIGURATION\n==========\n");

}

// TODO: option to print results
void parse_args(int argc, char *argv[]){
	int nrows_set = 0;
	int ncols_set = 0;
	int block_size_set = 0;
	int iter_set = 0;
	char c;
	while((c = getopt(argc, argv, "cgnmptbav")) != -1){
		if(c == '?'){
			printf("ERROR: unknown flag passed\n");
			exit(1);
		}
		switch(c){
		case 'c':
			SKIP_CPU = 1;
			break;
		case 'g':
			SKIP_CUDA = 1;
			break;
		case 'n':
			NROWS = atoi(argv[optind]);
			if(NROWS < 1){
				printf("ERROR: nrows should be at least 1\n");
				exit(1);
			}
			nrows_set = 1;
			break;
		case 'm':
			NCOLS = atoi(argv[optind]);
			if(NCOLS < 5){
				printf("ERROR: ncols should be at least 5\n");
				exit(1);
			}
			ncols_set = 1;
			break;

		case 'p':
			NUM_ITERATIONS = atoi(argv[optind]);
			iter_set = 1;
			break;
		case 't':
			PRINT_TIMING = 1;
			break;
		case 'b':
			BLOCK_SIZE = atoi(argv[optind]);
			block_size_set = 1;
			break;
		case 'a':
			AVERAGE_ROWS = 1;
			break;
		case 'v':
			PRINT_VALUES = 1;
			break;
		default: // should never get here
			break;
		}
	}
	print_config(nrows_set, ncols_set, block_size_set, iter_set	);
}

struct results_s {
	// CPU results
	long long cpu_time;
	DATA_TYPE **cpu_grid;
	DATA_TYPE *cpu_reduce;
	// GPU results for the naive version
	long long cuda_time_naive_ver;
	int cuda_naive_ver_matches;
	// GPU results for the optimised version
	long long cuda_time_fast_ver;
	int cuda_fast_ver_matches;
	// CPU results for the reduce
	long long cpu_time_reduce;
	// GPU results for the naive reduce
	long long cuda_time_reduce_naive_ver;
	int cuda_reduce_naive_ver_matches;
	// GPU results for the fast reduce
	long long cuda_time_reduce_fast_ver;
	int cuda_reduce_fast_ver_matches;
};

struct results_s results;

void print_result_check(){
	if(SKIP_CPU == 1 || SKIP_CUDA == 1){
		return; // skip comparison part
	}
	if(results.cuda_naive_ver_matches == 1){
		printf("ERROR: CPU and CUDA (naive version) grids do not match\n");
	} else {
		printf("SUCCESS: CPU and CUDA (naive version) grids match\n");
	}
	if(results.cuda_fast_ver_matches == 1){
		printf("ERROR: CPU and CUDA (fast version) grids do not match\n");
	} else {
		printf("SUCCESS: CPU and CUDA (fast version) grids match\n");
	}
	if(AVERAGE_ROWS){
		if(results.cuda_reduce_naive_ver_matches == 1){
			printf("ERROR: CPU and CUDA (naive version) reductions do not match\n");
		} else {
			printf("SUCCESS: CPU and CUDA (naive version) reductions match\n");
		}
		if(results.cuda_reduce_fast_ver_matches == 1){
			printf("ERROR: CPU and CUDA (fast version) reductions do not match\n");
		} else {
			printf("SUCCESS: CPU and CUDA (fast version) reductions match\n");
		}
	}
}

void print_timing(){
	if(PRINT_TIMING){
		printf("==========\nTIMING\n==========\n");
		if(SKIP_CPU == 0){
			printf("CPU took %lld microseconds\n", results.cpu_time);
			if(AVERAGE_ROWS){
				printf("CPU reduce took %lld microseconds\n", results.cpu_time_reduce);
			}
		}
		if(SKIP_CUDA == 0 && SKIP_CPU == 0){
			float ratio = (float) results.cpu_time / (float) results.cuda_time_naive_ver;
			printf("CUDA (naive version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_naive_ver, ratio);
			ratio = (float) results.cpu_time / (float) results.cuda_time_fast_ver;
			printf("CUDA (fast version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_fast_ver, ratio);
			if(AVERAGE_ROWS){
				ratio = (float) results.cpu_time_reduce / (float) results.cuda_time_reduce_naive_ver;
				printf("CUDA reduce (naive version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_reduce_naive_ver, ratio);
				ratio = (float) results.cpu_time_reduce / (float) results.cuda_time_reduce_fast_ver;
				printf("CUDA reduce (fast version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_reduce_fast_ver, ratio);
			}
		} else if(SKIP_CUDA == 0){
			printf("CUDA (naive version) took %lld microseconds\n", results.cuda_time_naive_ver);
			printf("CUDA (fast version) took %lld microseconds\n", results.cuda_time_fast_ver);
			if(AVERAGE_ROWS){
				printf("CUDA reduce (naive version) took %lld microseconds\n", results.cuda_time_reduce_naive_ver);
				printf("CUDA reduce (fast version) took %lld microseconds\n", results.cuda_time_reduce_fast_ver);
			}
		}
		printf("==========\nEND TIMING\n==========\n");
	}
}

void print_results(){
	print_timing();
	print_result_check();
}

// TODO: float/double weights
static DATA_TYPE **solve_system_cpu(){
	DATA_TYPE **cur = init_grid(NROWS, NCOLS);
	DATA_TYPE **next = init_grid(NROWS, NCOLS);
	int i;
	for(i = 0; i < NUM_ITERATIONS; i++){
		int n, m;
		for(n = 0; n < NROWS; n++){
			for(m = 2; m < NCOLS; m++){
				next[n][m] = 0.15*(cur[n][m-2]);
				next[n][m] += 0.65*(cur[n][m-1]);
				next[n][m] += (cur[n][m]);
				if(m == NCOLS - 2){
					next[n][m] += 1.35*(cur[n][m+1]);
					next[n][m] += 1.85*(cur[n][0]);
				} else if(m == NCOLS - 1){
					next[n][m] += 1.35*(cur[n][0]);
					next[n][m] += 1.85*(cur[n][1]);
				} else {
					next[n][m] += 1.35*(cur[n][m+1]);
					next[n][m] += 1.85*(cur[n][m+2]);
				}
				next[n][m] = next[n][m] / 5.0;
			}
		}
		DATA_TYPE **temp = cur;
		cur = next;
		next = temp;
	}
	free_grid(next);
	return cur;
}

void reduce_cpu(){
	results.cpu_reduce = calloc(NROWS, sizeof(DATA_TYPE));
	int i, n;
	for(i = 0; i < NROWS; i++){
		DATA_TYPE sum = 0.0;
		for(n = 0; n < NCOLS; n++){
			sum += results.cpu_grid[i][n];
		}
		results.cpu_reduce[i] = sum;
	}
}

void do_cpu_work(){
	struct timeval start, end;
	gettimeofday(&start, NULL);
	results.cpu_grid = solve_system_cpu();
	gettimeofday(&end, NULL);
	results.cpu_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_VALUES){
		printf("CPU grid:\n");
		print_grid(results.cpu_grid, NROWS, NCOLS);
	}
	if(AVERAGE_ROWS){
		gettimeofday(&start, NULL);
		reduce_cpu();
		gettimeofday(&end, NULL);
		results.cpu_time_reduce = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
		if(PRINT_VALUES){
			printf("CPU reduce:\n");
			print_reduce(results.cpu_reduce, NROWS);
		}
	}
}

void do_cuda_work(){
	struct timeval start, end;
	// First do naive version
	gettimeofday(&start, NULL);
	DATA_TYPE **naive_grid = init_grid(NROWS, NCOLS);
	do_grid_iterations_gpu_naive_ver(naive_grid, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS);
	gettimeofday(&end, NULL);
	results.cuda_time_naive_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(SKIP_CPU == 0){
		results.cuda_naive_ver_matches = compare_grids(results.cpu_grid, naive_grid, NROWS, NCOLS);
	}
	if(PRINT_VALUES){
		printf("CUDA grid (naive version):\n");
		print_grid(naive_grid, NROWS, NCOLS);
	}
	// TODO: reuse results below
	if(AVERAGE_ROWS){
		gettimeofday(&start, NULL);
		DATA_TYPE *naive_reduce = calloc(NROWS, sizeof(DATA_TYPE));
		do_reduce_naive(naive_grid, naive_reduce, NROWS, NCOLS, BLOCK_SIZE);
		gettimeofday(&end, NULL);
		results.cuda_time_reduce_naive_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
		if(SKIP_CPU == 0){
			results.cuda_reduce_naive_ver_matches = compare_reductions(results.cpu_reduce, naive_reduce, NROWS);
		}
		if(PRINT_VALUES){
			printf("CUDA reduce (naive version):\n");
			print_reduce(naive_reduce, NROWS);
		}
		free(naive_reduce);
	}
	free_grid(naive_grid);
	// Now do fast version
	gettimeofday(&start, NULL);
	DATA_TYPE **fast_grid = init_grid(NROWS, NCOLS);
	do_grid_iterations_gpu_fast_ver(fast_grid, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS);
	gettimeofday(&end, NULL);
	results.cuda_time_fast_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(SKIP_CPU == 0){
		results.cuda_fast_ver_matches = compare_grids(results.cpu_grid, fast_grid, NROWS, NCOLS);
	}
	if(PRINT_VALUES){
		printf("CUDA grid (fast version):\n");
		print_grid(fast_grid, NROWS, NCOLS);
	}
	// TODO: reuse results below
	if(AVERAGE_ROWS){
		gettimeofday(&start, NULL);
		DATA_TYPE *fast_reduce = calloc(NROWS, sizeof(DATA_TYPE));
		do_reduce_fast(fast_grid, fast_reduce, NROWS, NCOLS, BLOCK_SIZE);
		gettimeofday(&end, NULL);
		results.cuda_time_reduce_fast_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
		if(SKIP_CPU == 0){
			results.cuda_reduce_fast_ver_matches = compare_reductions(results.cpu_reduce, fast_reduce, NROWS);
		}
		if(PRINT_VALUES){
			printf("CUDA reduce (fast version):\n");
			print_reduce(fast_reduce, NROWS);
		}
		free(fast_reduce);
	}
	free_grid(fast_grid);
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	find_best_device();
	if(SKIP_CPU == 0){
		do_cpu_work();
	}
	if(SKIP_CUDA == 0){
		do_cuda_work();
	}
	print_results();
	return 0;
}

