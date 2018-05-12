OBJECTS = main.o cuda_code.cu
CC = g++
CFLAGS = -Wall -Wextra -std=c++11
NVCC = nvcc
NVCCFLAGS = -g -G --use_fast_math -std=c++11
INCPATH = -I. -I.
EXEC = exponentialIntegral.out

all: main.o cuda_code.o
	$(NVCC) -o ${EXEC} main.o cuda_code.o ${NVCCFLAGS}

main.o: main.cpp results.h
	$(CC) -c main.cpp $(CFLAGS)

cuda_code.o: cuda_code.cu results.h
	$(NVCC) -c cuda_code.cu $(NVCCFLAGS)

clean:
	rm -f *.o ${EXEC}
