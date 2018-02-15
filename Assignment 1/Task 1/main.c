#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include "matrix_funcs.h"

// These are the default values
static long long SEED = 123456L;
static int NROWS = 10;
static int NCOLS = 10;
static int PRINT_TIMING = 0;

void parse_args(int argc, char *argv[]){
	char c;
	while((c = getopt(argc, argv, "nmrt")) != -1){
		if(c == '?'){
			printf("ERROR: unknown flag passed\n");
			exit(1);			
		}
		switch(c){
		case 'n':
			NROWS = atoi(argv[optind]);
			break;
		case 'm':
			NCOLS = atoi(argv[optind]);
			break;
		case 'r': {
			struct timeval tv;
			gettimeofday(&tv, NULL);
			SEED = tv.tv_usec;
			break;
		}
		case 't':
			PRINT_TIMING = 1;
			break;
		default: // should never get here
			break;
		}
	}
}

void do_sums(float **mat){
	struct timeval start, end;
	// Time row summing
	gettimeofday(&start, NULL);
	float *row_vec = sum_rows_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	long long time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("Row summing took %lld microseconds\n", time_taken);
	}
	// Now time column summing
	gettimeofday(&start, NULL);
	float *col_vec = sum_cols_to_vector(mat, NROWS, NCOLS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("Column summing took %lld microseconds\n", time_taken);
	}
	// Time reducing the row vector
	gettimeofday(&start, NULL);
	float row_sum = reduce_vector(row_vec, NROWS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("Reducing row vector took %lld microseconds\n", time_taken);
	}
	// Time reducing the column vector
	gettimeofday(&start, NULL);
	float col_sum = reduce_vector(col_vec, NCOLS);
	gettimeofday(&end, NULL);
	time_taken = (end.tv_sec - start.tv_sec) * 1000000L + (end.tv_usec - start.tv_usec);
	if(PRINT_TIMING){
		printf("Reducing column vector took %lld microseconds\n", time_taken);
	}
	/*
	printf("Row sum vector is:\n");
	print_vector(row_vec, NROWS);
	printf("Row sum vector reduced is: %f\n", row_sum);
	printf("Column sum vector is:\n");
	print_vector(col_vec, NCOLS);
	printf("Column sum vector reduced is: %f\n", col_sum);
	*/
	printf("Row sum vector reduced is: %f\n", row_sum);
	printf("Column sum vector reduced is: %f\n", col_sum);
	free(row_vec);
	free(col_vec);
}

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	srand48(SEED);
	float **mat = create_random_matrix(NROWS, NCOLS);
	//print_matrix(mat, NROWS, NCOLS);
	do_sums(mat);
	free_matrix(mat);
	return 0;
}
