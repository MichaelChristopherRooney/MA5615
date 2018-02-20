OBJECTS = main.o matrix.o
EXEC_NAME = prog
NVCC = nvcc
CFLAGS = -Wall -lm -Wextra
NVCCFLAGS = -g -G --use_fast_math

all: main.o matrix.o
	$(NVCC) -o $(EXEC_NAME) matrix.o main.o $(NVCCFLAGS)

main.o: matrix.h main.cu
	$(NVCC) -c main.cu $(NVCCFLAGS)

matrix.o: matrix.cu matrix.h
	$(NVCC) -c matrix.cu $(NVCCFLAGS)

clean:
	rm -f $(OBJECTS) $(EXEC_NAME)

