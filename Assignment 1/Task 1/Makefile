OBJECTS = main.o matrix_funcs.o
EXEC_NAME = prog
CC = gcc
CFLAGS = -Wall -lm -Wextra

all: main.o matrix_funcs.o
	$(CC) -o $(EXEC_NAME) main.o matrix_funcs.o $(CFLAGS)

main.o: main.c matrix_funcs.h
	$(CC) -c main.c $(CFLAGS)

matrix_funcs.o: matrix_funcs.c matrix_funcs.h
	$(CC) -c matrix_funcs.c $(CFLAGS)

clean:
	rm -f $(OBJECTS) $(EXEC_NAME)

