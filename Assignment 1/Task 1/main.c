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

int main(int argc, char *argv[]){
	parse_args(argc, argv);
	srand48(SEED);
	printf("NROWS: %d\n", NROWS);
	printf("NCOLS: %d\n", NCOLS);
	float **mat = create_random_matrix(NROWS, NCOLS);
	print_matrix(mat, NROWS, NCOLS);
	free_matrix(mat);
	return 0;
}
