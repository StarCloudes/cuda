# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
PARALLEL	= -lcurand
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
#NVCCFLAGS	= -g -G --use_fast_math
NVCCFLAGS	= -O4 --use_fast_math -lcurand

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= thrustRandomVectorsComparison.cu
OBJECTS		= thrustRandomVectorsComparison.o

TARGET 		= thrustRandomVectorsComparison


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o thrustRandomVectorsComparison -I$(INCPATH) $(NVCCFLAGS)

thrustRandomVectorsComparison.o: thrustRandomVectorsComparison.cu
	$(NVCC)  thrustRandomVectorsComparison.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
