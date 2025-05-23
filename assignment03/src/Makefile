NVCC        = nvcc
CXX         = g++
CXXFLAGS    = -O3 -std=c++11
NVCCFLAGS   = -O3 -arch=sm_60

INCLUDES    = -I.

OBJS_CPP    = main.o
OBJS_CU     = exponentialIntegral_gpu.o

EXEC        = exponentialIntegral.out

all: $(EXEC)

$(EXEC): $(OBJS_CPP) $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f *.o $(EXEC)