#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "grid.h"
#include "data_type.h"

// CUDA functions from gpu_code.cu
extern void find_best_device();
extern DATA_TYPE *do_grid_iterations_gpu_naive_ver(
                DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size,
                int num_iter, float *time_alloc, float *copy_to, float *time_exec, 
		float *time_copy_from);
extern DATA_TYPE *do_grid_iterations_gpu_fast_ver(
		DATA_TYPE **grid_gpu_host, int nrow, int ncol, int block_size, 
		int num_iter, float *time_alloc, float *time_exec, float *time_copy_from);
extern void do_reduce_cuda(
		DATA_TYPE *grid_device, DATA_TYPE *reduce_host, int nrow, int ncol, 
		int block_size, float *time_alloc, float *time_exec, float *time_copy_from);
extern void free_on_device(DATA_TYPE *device_ptr);

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
			if(NROWS % 32 != 0){
				printf("ERROR: nrows should be a multiple of 32\n");
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
			if(NCOLS % 32 != 0){
				printf("ERROR: ncols should be a multiple of 32\n");
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
	float cuda_time_naive_alloc;
	float cuda_time_naive_copy_to;
	float cuda_time_naive_exec;
	float cuda_time_naive_copy_from;
	// GPU results for the optimised version
	long long cuda_time_fast_ver;
	int cuda_fast_ver_matches;
	float cuda_time_fast_alloc;
	float cuda_time_fast_exec;
	float cuda_time_fast_copy_from;
	// CPU results for the reduce
	long long cpu_time_reduce;
	// GPU results for the reduce
	long long cuda_time_reduce;
	int cuda_reduce_matches;
	float cuda_reduce_time_alloc;
	float cuda_reduce_time_exec;
	float cuda_reduce_time_copy_from;
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
		if(results.cuda_reduce_matches == 1){
			printf("ERROR: CPU and CUDA reductions do not match\n");
		} else {
			printf("SUCCESS: CPU and CUDA reductions match\n");
		}
	}
}

void print_fast_timing(){
	if(SKIP_CUDA == 1){
		return;
	}
	if(SKIP_CPU == 0){
		float ratio = (float) results.cpu_time / (float) results.cuda_time_fast_ver;
		printf("CUDA FAST: Took %lld milliseconds overall excluding reduce (%fx faster than CPU)\n", results.cuda_time_fast_ver, ratio);
	} else {
		printf("CUDA FAST: Took %lld milliseconds overall excluding reduce\n", results.cuda_time_fast_ver);
	}
	printf("CUDA FAST: Allocating one device buffer took %f milliseconds\n", results.cuda_time_fast_alloc);
	printf("CUDA FAST: No copy to device needed - no time taken\n");
	printf("CUDA FAST: Execution of fast kernel took %f milliseconds\n", results.cuda_time_fast_exec);
	printf("CUDA FAST: Copying buffer from device took %f milliseconds\n", results.cuda_time_fast_copy_from);
}

void print_naive_timing(){
	if(SKIP_CUDA == 1){
		return;
	}
	if(SKIP_CPU == 0){
		float ratio = (float) results.cpu_time / (float) results.cuda_time_naive_ver;
		printf("CUDA NAIVE: Took %lld milliseconds overall excluding reduce (%fx faster than CPU)\n", results.cuda_time_naive_ver, ratio);
	} else {
		printf("CUDA NAIVE: Took %lld milliseconds overall excluding reduce\n", results.cuda_time_naive_ver);
	}
	printf("CUDA NAIVE: Allocating two device buffers took %f milliseconds\n", results.cuda_time_naive_alloc);
	printf("CUDA NAIVE: Copying two buffers to device took %f milliseconds\n", results.cuda_time_naive_copy_to);
	printf("CUDA NAIVE: Execution of naive kernel took %f milliseconds\n", results.cuda_time_naive_exec);
	printf("CUDA NAIVE: Copying one buffer from device took %f milliseconds\n", results.cuda_time_naive_copy_from);
}

void print_cuda_reduce_timing(){
	if(AVERAGE_ROWS && SKIP_CUDA == 0){
		if(SKIP_CPU == 0){
			float ratio = (float) results.cpu_time_reduce / (float) results.cuda_time_reduce;
			printf("CUDA REDUCE: Took %lld milliseconds overall (%fx faster than CPU)\n", results.cuda_time_reduce, ratio);
		} else {
			printf("CUDA REDUCE: Took %lld milliseconds overall\n", results.cuda_time_reduce);
		}
		printf("CUDA REDUCE: Allocating device buffer took %f milliseconds\n", results.cuda_reduce_time_alloc);
		printf("CUDA REDUCE: No copy to device needed - no time taken\n");
		printf("CUDA REDUCE: Execution of reduce kernel took %f milliseconds\n", results.cuda_reduce_time_exec);
		printf("CUDA REDUCE: Copying buffer from device took %f milliseconds\n", results.cuda_reduce_time_copy_from);
	}
}

void print_cpu_timing(){
	if(SKIP_CPU == 0){
		printf("CPU: Took %lld milliseconds overall excluding reduce\n", results.cpu_time);
		if(AVERAGE_ROWS){
			printf("CPU: Reduce took %lld milliseconds\n", results.cpu_time_reduce);
		}
	}
}

void print_timing(){
	if(PRINT_TIMING){
		printf("==========\nTIMING\n==========\n");
		print_cpu_timing();
		print_naive_timing();
		print_fast_timing();
		print_cuda_reduce_timing();
		printf("==========\nEND TIMING\n==========\n");
	}
}

void print_results(){
	print_timing();
	print_result_check();
}

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
		results.cpu_reduce[i] = sum / (DATA_TYPE) NCOLS;
	}
}

void do_cpu_work(){
	struct timeval start, end;
	gettimeofday(&start, NULL);
	results.cpu_grid = solve_system_cpu();
	gettimeofday(&end, NULL);
	results.cpu_time = ((end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec)) / 1000;
	if(PRINT_VALUES){
		printf("CPU grid:\n");
		print_grid(results.cpu_grid, NROWS, NCOLS);
	}
	if(AVERAGE_ROWS){
		gettimeofday(&start, NULL);
		reduce_cpu();
		gettimeofday(&end, NULL);
		results.cpu_time_reduce = ((end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec)) / 1000;
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
	DATA_TYPE *grid_device = do_grid_iterations_gpu_naive_ver(
		naive_grid, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS,
		&results.cuda_time_naive_alloc, &results.cuda_time_naive_copy_to,
		&results.cuda_time_naive_exec, &results.cuda_time_naive_copy_from
	);
	gettimeofday(&end, NULL);
	results.cuda_time_naive_ver = ((end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec)) / 1000;
	if(SKIP_CPU == 0){
		results.cuda_naive_ver_matches = compare_grids(results.cpu_grid, naive_grid, NROWS, NCOLS);
	}
	if(PRINT_VALUES){
		printf("CUDA grid (naive version):\n");
		print_grid(naive_grid, NROWS, NCOLS);
	}
	free_on_device(grid_device);
	free_grid(naive_grid);
	// Now do fast version
	gettimeofday(&start, NULL);
	DATA_TYPE **fast_grid = init_grid(NROWS, NCOLS);
	grid_device = do_grid_iterations_gpu_fast_ver(
		fast_grid, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS,
		&results.cuda_time_fast_alloc, &results.cuda_time_fast_exec,
		&results.cuda_time_fast_copy_from
	);
	gettimeofday(&end, NULL);
	results.cuda_time_fast_ver = ((end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec)) / 1000;
	if(SKIP_CPU == 0){
		results.cuda_fast_ver_matches = compare_grids(results.cpu_grid, fast_grid, NROWS, NCOLS);
	}
	if(PRINT_VALUES){
		printf("CUDA grid (fast version):\n");
		print_grid(fast_grid, NROWS, NCOLS);
	}
	// Reduce reuses values from the fast version's grid
	if(AVERAGE_ROWS){
		gettimeofday(&start, NULL);
		DATA_TYPE *cuda_reduce = calloc(NROWS, sizeof(DATA_TYPE));
		do_reduce_cuda(
			grid_device, cuda_reduce, NROWS, NCOLS, BLOCK_SIZE, 
			&results.cuda_reduce_time_alloc, &results.cuda_reduce_time_exec,
			&results.cuda_reduce_time_copy_from
		);
		gettimeofday(&end, NULL);
		results.cuda_time_reduce = ((end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec)) / 1000;
		if(SKIP_CPU == 0){
			results.cuda_reduce_matches = compare_reductions(results.cpu_reduce, cuda_reduce, NROWS);
		}
		if(PRINT_VALUES){
			printf("CUDA reduce:\n");
			print_reduce(cuda_reduce, NROWS);
		}
		free(cuda_reduce);
	}
	free_on_device(grid_device);
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

