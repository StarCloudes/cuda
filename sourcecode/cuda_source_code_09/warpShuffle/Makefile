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
NVCCFLAGS	= -g -G --use_fast_math -gencode arch=compute_30,code=sm_30
# To use the warp match functions, you will need to use the "-arch=sm_70" flag - and a GPU which is CC 7.0 or more
#NVCCFLAGS	= -g -G --use_fast_math -arch=sm_70
# Another option would be to use "-gencode arch=compute_70,code=sm_70"
#NVCCFLAGS	= -g -G --use_fast_math -gencode arch=compute_70,code=sm_70

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= warpShuffle.cu
OBJECTS		= warpShuffle.o

TARGET 		= warpShuffle


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o warpShuffle -I$(INCPATH)

warpShuffle.o: warpShuffle.cu
	$(NVCC)  warpShuffle.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS)
