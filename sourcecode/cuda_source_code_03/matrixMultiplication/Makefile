# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -O4 -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -O4 -W -Wall $(PARALLEL) $(DEFINES)
#NVCCFLAGS	= -g -G --use_fast_mathNVCCFLAGS
NVCCFLAGS       = -O4 --use_fast_math --compiler-options -fno-inline


LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= matrixMultiplication.cu
OBJECTS		= matrixMultiplication.o

TARGET 		= matrixMultiplication


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o matrixMultiplication $(LIBS)-I$(INCPATH)

matrixMultiplication.o: matrixMultiplication.cu
	$(NVCC)  matrixMultiplication.cu -c $(NVCCFLAGS) -I$(INCPATH)

test: matrixMultiplication
	./matrixMultiplication -n 5000 -m 5000 >output_05000.txt
	./matrixMultiplication -n 6000 -m 6000 >output_06000.txt
	./matrixMultiplication -n 7000 -m 7000 >output_07000.txt
	./matrixMultiplication -n 8000 -m 8000 >output_08000.txt
	./matrixMultiplication -n 9000 -m 9000 >output_09000.txt
	./matrixMultiplication -n 10000 -m 10000 >output_10000.txt
	./matrixMultiplication -n 11000 -m 11000 >output_11000.txt
	./matrixMultiplication -n 12000 -m 12000 >output_12000.txt
	./matrixMultiplication -n 13000 -m 13000 >output_13000.txt
clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
