OBJECTS = main.o matrix.o gpu_code.o
EXEC_NAME = prog
NVCC = nvcc
CFLAGS = -Wall -lm -Wextra -D DATA_TYPE=$(DATA_TYPE_USED)
NVCCFLAGS = -g -G --use_fast_math -D DATA_TYPE=$(DATA_TYPE_USED)

all: main.o matrix.o gpu_code.o
	$(NVCC) -o $(EXEC_NAME) gpu_code.o matrix.o main.o $(NVCCFLAGS)

main.o: matrix.h main.c
	$(CC) -c main.c $(CFLAGS)

matrix.o: matrix.c matrix.h
	$(CC) -c matrix.c $(CFLAGS)

gpu_code.o: gpu_code.cu matrix.h
	$(NVCC) -c gpu_code.cu $(NVCCFLAGS)

clean:
	rm -f $(OBJECTS) $(EXEC_NAME)


