OBJECTS = main.o cuda_code.cu
CC = g++
CFLAGS = -Wall -Wextra
NVCC = nvcc
NVCCFLAGS = -g -G --use_fast_math
INCPATH = -I. -I.
EXEC = exponentialIntegral.out

all: main.o cuda_code.o
	$(NVCC) -o ${EXEC} main.o cuda_code.o ${NVCCFLAGS}

main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS)

cuda_code.o: cuda_code.cu
	$(NVCC) -c cuda_code.cu $(NVCCFLAGS)

clean:
	rm -f *.o ${EXEC}
