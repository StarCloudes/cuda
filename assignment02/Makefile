# Makefile
NVCC = nvcc
SRC = src/main.cu src/heat_gpu.cu src/heat_cpu.cpp
TARGET = heat_gpu

INCLUDES = -Iinclude

all:
	$(NVCC) $(INCLUDES) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)