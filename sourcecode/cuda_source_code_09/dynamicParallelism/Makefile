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
NVCCFLAGS	= -g -G --use_fast_math -arch=sm_35 --relocatable-device-code true -lcudadevrt

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= dynamicParalellism.cu
OBJECTS		= dynamicParalellism.o

TARGET 		= dynamicParalellism


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o dynamicParalellism -I$(INCPATH) -lcudadevrt $(NVCCFLAGS)

dynamicParalellism.o: dynamicParalellism.cu
	$(NVCC)  dynamicParalellism.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS)
