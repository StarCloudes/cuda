CC=g++
NVCC=		nvcc
LINK=		nvcc

#CFLAGS= -g -pedantic -W -Wall -L/usr/lib
#CFLAGS= -O2
CFLAGS= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#NVCCFLAGS	= -g -G --use_fast_math -arch=sm_30
#NVCCFLAGS	= -g -G --use_fast_math -arch=sm_30 -DWITH_MY_DEBUG
NVCCFLAGS	= -O2 --use_fast_math -arch=sm_30

INCPATH       = -I. -I.

SOURCES		=	matrixMultiplicationCommented.cu \
				cudaUtils.cu

OBJECTS		=	matrixMultiplicationCommented.o \
				cudaUtils.o \
		  		main.o

TARGET 		= matrixMultiplicationCommented.out

all: main

main: $(OBJECTS)
#	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lefence
	$(NVCC) $(OBJECTS) -o $(TARGET) $(NVCCFLAGS) -I$(INCPATH) 

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $(INCPATH) $<

matrixMultiplicationCommented.o: matrixMultiplicationCommented.cu
	$(NVCC)  matrixMultiplicationCommented.cu -c $(NVCCFLAGS) -I$(INCPATH)
cudaUtils.o: cudaUtils.cu
	$(NVCC)  cudaUtils.cu -c $(NVCCFLAGS) -I$(INCPATH)

install:

clean:
	rm -f *.o  ${TARGET}

