BLOCK_X ?= 16
BLOCK_Y ?= 16
CUDACFLAGS  = -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)
# Single precision
heat_sim: CUDACFLAGS += 
heat_sim: INCLUDES += -Iinclude
heat_sim: SRCS = src/main.cu src/heat_gpu.cu src/heat_cpu.cpp
heat_sim:
	nvcc $(INCLUDES) $(CUDACFLAGS) -o $@ $(SRCS)

# Double precision
heat_sim_dp: CUDACFLAGS += -DUSE_DOUBLE
heat_sim_dp: INCLUDES += -Iinclude
heat_sim_dp: SRCS = src/main.cu src/heat_gpu.cu src/heat_cpu.cpp
heat_sim_dp:
	nvcc $(INCLUDES) $(CUDACFLAGS) -o $@ $(SRCS)

.PHONY: heat_sim heat_sim_dp clean
clean:
	rm -f heat_sim heat_sim_dp temp_output.txt