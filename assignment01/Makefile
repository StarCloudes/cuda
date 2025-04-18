###############################################################################
# Makefile that builds three separate executables:
#   1) task01_matrix_sums  from task01_matrix_sums.cc     (CPU/C++ code)
#   2) task02_matrix_sums  from task02_matrix_sums_cuda.cu (CUDA code)
#   3) task03_matrix_sums  from task03_matrix_sums_cuda.cu (CUDA code)
#   4) task04_matrix_sums  from task04_matrix_sums_cuda.cu (CUDA code)
#
# Usage:
#   make          # Build all three
#   make clean    # Remove all executables
###############################################################################

# Compiler commands
CXX  = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS  = -O2 -std=c++11
NVCCFLAGS = -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75

# Executable names
CPU_EXE  = task01_matrix_sums
CUDA_EXE2 = task02_matrix_sums
CUDA_EXE3 = task03_matrix_sums
CUDA_EXE4 = task04_matrix_sums

# Default target: build all
all: $(CPU_EXE) $(CUDA_EXE2) $(CUDA_EXE3) $(CUDA_EXE4)

###############################################################################
# 1) Build CPU executable from task01_matrix_sums.cc
###############################################################################
$(CPU_EXE): task01_matrix_sums.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

###############################################################################
# 2) Build CUDA executable from task02_matrix_sums_cuda.cu
###############################################################################
$(CUDA_EXE2): task02_matrix_sums_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

###############################################################################
# 3) Build CUDA executable from task03_matrix_sums_cuda.cu
###############################################################################
$(CUDA_EXE3): task03_matrix_sums_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

###############################################################################
# 4) Build CUDA executable from task03_matrix_sums_cuda.cu
###############################################################################
$(CUDA_EXE4): task04_matrix_sums_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<


###############################################################################
# 5) Runing tasks
###############################################################################
run_task01:
	./$(CPU_EXE) -n 5000 -m 5000 -t

run_task02:
	/usr/local/cuda-12.6/bin/compute-sanitizer ./$(CUDA_EXE2) -n 5000 -m 5000 -t -gt

run_task03:
	/usr/local/cuda-12.6/bin/compute-sanitizer ./$(CUDA_EXE3) -auto -t -gt -sp -save

run_task04:
	/usr/local/cuda-12.6/bin/compute-sanitizer ./$(CUDA_EXE4) -auto -t -gt -sp -save

###############################################################################
# Clean up all generated executables (and .o files if any)
###############################################################################
clean:
	rm -f $(CPU_EXE) $(CUDA_EXE2) $(CUDA_EXE3) $(CUDA_EXE4) *.o