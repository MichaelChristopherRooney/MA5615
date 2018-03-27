OBJECTS = main.o grid.o gpu_code.o
EXEC_NAME = prog
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -lm -Wextra -D DATA_TYPE=$(DATA_TYPE_USED)
NVCCFLAGS = -g -G --use_fast_math -D DATA_TYPE=$(DATA_TYPE_USED)

all: main.o grid.o gpu_code.o
	$(NVCC) -o $(EXEC_NAME) gpu_code.o grid.o main.o $(NVCCFLAGS)

main.o: grid.h main.c
	$(CC) -c main.c $(CFLAGS)

grid.o: grid.c grid.h
	$(CC) -c grid.c $(CFLAGS)

gpu_code.o: gpu_code.cu grid.h
	$(NVCC) -c gpu_code.cu $(NVCCFLAGS)

clean:
	rm -f $(OBJECTS) $(EXEC_NAME)


