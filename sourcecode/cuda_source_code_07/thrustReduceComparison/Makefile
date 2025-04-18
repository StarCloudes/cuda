# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
#NVCCFLAGS	= -g -G --use_fast_math -arch=sm_21
NVCCFLAGS	= -O4 --use_fast_math -arch=sm_30

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= thrustReduceComparison.cu
OBJECTS		= thrustReduceComparison.o

TARGET 		= thrustReduceComparison


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o thrustReduceComparison -I$(INCPATH)

thrustReduceComparison.o: thrustReduceComparison.cu
	$(NVCC)  thrustReduceComparison.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)

test: thrustReduceComparison
	./thrustReduceComparison 0
	./thrustReduceComparison 1
	./thrustReduceComparison 2
	./thrustReduceComparison 3
	./thrustReduceComparison 4
	./thrustReduceComparison 5
	./thrustReduceComparison 6
