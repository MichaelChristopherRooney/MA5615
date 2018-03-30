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

// These are the default values
static int NROWS = 32;
static int NCOLS = 32;
static int NUM_ITERATIONS = 10;
static int PRINT_TIMING = 0;
static int PRINT_VALUES = 0;
static int AVERAGE_ROWS = 0;
static int SKIP_CPU = 0;
static int SKIP_CUDA = 0;
int BLOCK_SIZE = 32;

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
	DATA_TYPE *cpu_row_averages;
	// GPU results for the version that only uses global memory
	long long cuda_time_naive_ver;
	DATA_TYPE **cuda_grid_naive_ver;
	// GPU results for the optimised version
	long long cuda_time_fast_ver;
	DATA_TYPE **cuda_grid_fast_ver;
};

struct results_s results;

void print_results(){
	if(PRINT_TIMING){
		printf("==========\nTIMING\n==========\n");
		if(SKIP_CPU == 0){
			printf("CPU took %lld microseconds\n", results.cpu_time);
		}
		if(SKIP_CUDA == 0 && SKIP_CPU == 0){
			float ratio = (float) results.cpu_time / (float) results.cuda_time_naive_ver;
			printf("CUDA (naive version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_naive_ver, ratio);
			ratio = (float) results.cpu_time / (float) results.cuda_time_fast_ver;
			printf("CUDA (fast version) took %lld microseconds (%fx faster than CPU)\n", results.cuda_time_fast_ver, ratio);
		} else if(SKIP_CUDA == 0){
			printf("CUDA (naive version) took %lld microseconds\n", results.cuda_time_naive_ver);
			printf("CUDA (fast version) took %lld microseconds\n", results.cuda_time_fast_ver);
		}
		printf("==========\nEND TIMING\n==========\n");
	}
	if(PRINT_VALUES){
		printf("==========\nGRID\n==========\n");
		if(SKIP_CPU == 0){
			printf("CPU grid:\n");
			print_grid(results.cpu_grid, NROWS, NCOLS);
		}
		if(SKIP_CUDA == 0){
			printf("CUDA grid (naive version):\n");
			print_grid(results.cuda_grid_naive_ver, NROWS, NCOLS);
			printf("CUDA grid (fast version):\n");
			print_grid(results.cuda_grid_fast_ver, NROWS, NCOLS);
		}
		printf("==========\nEND GRID\n==========\n");
	}
	if(SKIP_CPU == 1 || SKIP_CUDA == 1){
		return; // skip comparison part
	}
	// TODO: compare other CUDA versions
	int res = compare_grids(results.cpu_grid, results.cuda_grid_naive_ver, NROWS, NCOLS);
	if(res == 1){
		printf("ERROR: CPU and CUDA (naive version) grids do not match\n");
	} else {
		printf("SUCCESS: CPU and CUDA (naive version) grids match\n");
	}
	res = compare_grids(results.cpu_grid, results.cuda_grid_fast_ver, NROWS, NCOLS);
	if(res == 1){
		printf("ERROR: CPU and CUDA (fast version) grids do not match\n");
	} else {
		printf("SUCCESS: CPU and CUDA (fast version) grids match\n");
	}
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
	if(AVERAGE_ROWS){
		printf("TODO: average CPU row results\n");
	}
	return cur;
}

void do_work(){
	struct timeval start, end;
	if(SKIP_CPU == 0){
		gettimeofday(&start, NULL);
		results.cpu_grid = solve_system_cpu();
		gettimeofday(&end, NULL);
		results.cpu_time = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	}
	if(SKIP_CUDA == 0){
		// First do global memory version
		gettimeofday(&start, NULL);
		results.cuda_grid_naive_ver = init_grid(NROWS, NCOLS);
		do_grid_iterations_gpu_naive_ver(results.cuda_grid_naive_ver, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS);
		gettimeofday(&end, NULL);
		results.cuda_time_naive_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
		sleep(1); // TODO: 
		// Now do global memory and reg version
		gettimeofday(&start, NULL);
		results.cuda_grid_fast_ver = init_grid(NROWS, NCOLS);
		do_grid_iterations_gpu_fast_ver(results.cuda_grid_fast_ver, NROWS, NCOLS, BLOCK_SIZE, NUM_ITERATIONS);
		gettimeofday(&end, NULL);
		results.cuda_time_fast_ver = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	}
	// Now print results and cleanup
	print_results();
	if(SKIP_CUDA == 0){
		// TODO: clean up
	}
	if(SKIP_CPU == 0){
		// TODO: clean up
	}
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	find_best_device();
	do_work();
	return 0;
}

