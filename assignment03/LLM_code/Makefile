# Makefile for CUDA Exponential Integral calculation
# MAP55616-03 assignment

# Compiler settings
NVCC = nvcc
CXX = g++

# Compiler flags
NVCCFLAGS = -O3  -std=c++11
CXXFLAGS = -O3 -std=c++11


# Directories
SRCDIR = .
OBJDIR = obj
BINDIR = bin

# Source files
CUDA_SOURCES = exponential_integral_cuda.cu
CPP_SOURCES = main.cpp
HEADERS = exponential_integral_cuda.h

# Object files
CUDA_OBJECTS = $(OBJDIR)/exponential_integral_cuda.o
CPP_OBJECTS = $(OBJDIR)/main.o

# Target executable
TARGET = $(BINDIR)/exponentialIntegral

# Libraries
LIBS = -lcudart

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

# Link the final executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS)  -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS)  -c $< -o $@

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS)  -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "Clean complete"

# Install (copy to system path - optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	@echo "Installation complete"

# Run tests with different parameters
test: $(TARGET)
	@echo "Running basic test..."
	./$(TARGET) -t -v -n 5 -m 5
	@echo "\nRunning performance test..."
	./$(TARGET) -t -n 10 -m 100
	@echo "\nRunning large scale test..."
	./$(TARGET) -t -n 20 -m 1000

# Run only GPU version
test-gpu: $(TARGET)
	@echo "Running GPU-only test..."
	./$(TARGET) -c -t -v -n 10 -m 50

# Run only CPU version  
test-cpu: $(TARGET)
	@echo "Running CPU-only test..."
	./$(TARGET) -g -t -v -n 10 -m 50

# Debug build
debug: NVCCFLAGS += -g -G -DDEBUG
debug: CXXFLAGS += -g -DDEBUG
debug: directories $(TARGET)

# Performance analysis build
profile: NVCCFLAGS += -lineinfo
profile: directories $(TARGET)

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || echo "NVCC not found!"
	@nvidia-smi || echo "NVIDIA driver not found!"

# Show GPU information
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi -L
	@echo "\nDetailed GPU Info:"
	@nvidia-smi

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  clean     - Remove build files"
	@echo "  test      - Run basic tests"
	@echo "  test-gpu  - Run GPU-only tests"
	@echo "  test-cpu  - Run CPU-only tests"
	@echo "  debug     - Build with debug information"
	@echo "  profile   - Build with profiling information"
	@echo "  check-cuda- Check CUDA installation"
	@echo "  gpu-info  - Show GPU information"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  make test"
	@echo "  make CUDA_ARCH=sm_70"
	@echo "  ./bin/exponentialIntegral -t -v -n 10 -m 100"

.PHONY: all clean test test-gpu test-cpu debug profile check-cuda gpu-info help directories install