CC=g++
CFLAGS= -g -pedantic -W -Wall -L/usr/lib
#CFLAGS= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

INCPATH       = -I. -I.

TARGET=main.o
EXEC=exponentialIntegral.out


all: $(TARGET)
#	$(CC) -Wall -o ${EXEC} ${TARGET}
	$(CC) -Wall -o ${EXEC} ${TARGET} -lefence
	

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $(INCPATH) $<
	
install:

clean:
	rm -f *.o ${TARGET}
