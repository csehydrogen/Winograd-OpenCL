CPPFLAGS=-I/usr/local/cuda/include
CFLAGS=-std=gnu99
LDLIBS=-lOpenCL

all: main

main: timer.o

clean:
	rm -rf *.o main
