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
NVCCFLAGS	= -g -G --use_fast_math -arch=sm_30

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/:/usr/include/linux/:/usr/local/cuda/include
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= atomicMaxMinDouble.cu
OBJECTS		= atomicMaxMinDouble.o

TARGET 		= atomicMaxMinDouble


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o atomicMaxMinDouble.out -I$(INCPATH)

atomicMaxMinDouble.o: atomicMaxMinDouble.cu
	$(NVCC)  atomicMaxMinDouble.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) *.out
