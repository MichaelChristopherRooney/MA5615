OBJECTS = main.o grid.o gpu_code.o
EXEC_NAME = prog
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -lm -Wextra -D USE_FLOAT=$(USES_FLOAT)
NVCCFLAGS = -g -G --use_fast_math -D USE_FLOAT=$(USES_FLOAT)

all: main.o grid.o gpu_code.o
	$(NVCC) -o $(EXEC_NAME) gpu_code.o grid.o main.o $(NVCCFLAGS)

main.o: grid.h data_type.h main.c
	$(CC) -c main.c $(CFLAGS)

grid.o: grid.c grid.h data_type.h
	$(CC) -c grid.c $(CFLAGS)

gpu_code.o: gpu_code.cu grid.h data_type.h
	$(NVCC) -c gpu_code.cu $(NVCCFLAGS)

clean:
	rm -f $(OBJECTS) $(EXEC_NAME)


