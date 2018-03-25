#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "grid.h"

//extern void find_best_device();

// These are the default values
static int NROWS = 32;
static int NCOLS = 32;
static int NUM_ITERATIONS = 10;
static int PRINT_TIMING = 0;
static int PRINT_VALUES = 0;
static DATA_TYPE PRECISION = 1.E-5;
static int AVERAGE_ROWS = 0;
static int SKIP_CPU = 0;
static int SKIP_CUDA = 0;
int BLOCK_SIZE = 8;

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
	while((c = getopt(argc, argv, "cgnmptba")) != -1){
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
			nrows_set = 1;
			break;
		case 'm':
			NCOLS = atoi(argv[optind]);
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
	// GPU results
	long long cuda_time;
};

struct results_s results;

void print_results(DATA_TYPE **grid){
	if(PRINT_TIMING){
		printf("==========\nTIMING\n==========\n");
		if(SKIP_CPU == 0){
			// TODO: CPU timing results
		}
		if(SKIP_CUDA == 0){
			// TODO: CUDA timing results
		}
		printf("==========\nEND TIMING\n==========\n");
	}
	if(PRINT_VALUES){
		printf("==========\nGRID\n==========\n");
		if(SKIP_CPU == 0){
			// TODO: print CPU grid
		}
		if(SKIP_CUDA == 0){
			// TODO: print GPU grid
		}
		printf("==========\nEND GRID\n==========\n");
	}
	if(SKIP_CPU == 1 || SKIP_CUDA == 1){
		return; // skip comparison part
	}
	// TODO: compare results
}

// TODO: float/double weights
// TODO: timing here
static void solve_system_cpu(){
	DATA_TYPE **cur = init_grid(NROWS, NCOLS);
	DATA_TYPE **next = init_grid(NROWS, NCOLS);
	//print_grid(cur, NROWS, NCOLS);
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
		//printf("\n\n\nIteration: %d\n", i);
		//print_grid(cur, NROWS, NCOLS);
	}
	free_grid(next);
	results.cpu_grid = cur;
	if(AVERAGE_ROWS){
		printf("TODO: average CPU row results\n");
	}
	//printf("\n\nResults:\n");
	//print_grid(cur, NROWS, NCOLS);

}

void time_work(){
	struct timeval start, end;
	if(SKIP_CPU == 0){
		solve_system_cpu();
	}
	if(SKIP_CUDA == 0){
		// TODO: CUDA version
	}
	// Now print results and cleanup
	//print_results(grid, results);
	if(SKIP_CUDA == 0){
		// TODO: clean up
	}
	if(SKIP_CPU == 0){
		// TODO: clean up
	}
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	//find_best_device();
	time_work();
	//free_grid(grid);
	return 0;
}

