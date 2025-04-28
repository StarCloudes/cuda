# 🧊 Heat Simulation - Assignment 02
This project implements a 1D heat propagation simulation using both CPU and GPU (CUDA), with optional timing, verification, and row-averaging features.

## 📁 File Structure

**plot/plot_speedup.py** (Plotting with Python script.)

**src/main.cu** (Main program entrypoint. Handles argument parsing, memory management, CPU and GPU orchestration.)

**src/heat_cpu.h/.cpp** (CPU-side heat propagation and row average functions.)

**src/heat_gpu.cuh/.cu** (GPU kernel declarations and implementations.)

**src/real.h** (Real_t typedef (float vs double).)

**run_test.sh** (Automated perf/accuracy test script)

**Writeup.pdf** (Report)

**Makefile** 


## 🚀 Build Instructions

To compile the project, run: 

### Single-precision (default)
```
make heat_sim
```

### Double-precision 
```
make heat_sim_dp
```

### Override block dimensions at compile time with:
```
make heat_sim_dp BLOCK_X=32 BLOCK_Y=8
```

## 🧪 Usage
```
./heat_sim [options]
./heat_sim -n 4 -m 128 -p 5 --cpu-only
./heat_sim -n 256 -m 256 -p 5 -g

```

## 📊 Automated Testing Script

Use run_test.sh to sweep matrix sizes and thread‐block configurations automatically:
```
chmod +x run_test.sh
./run_test.sh
```

## ✅Options

 Option         | Description                                      |
|----------------|--------------------------------------------------|
| `-n <rows>`    | Number of matrix rows (default: 32)              |
| `-m <cols>`    | Number of matrix columns (default: 32)           |
| `-p <iters>`   | Number of simulation iterations (default: 10)    |
| `-a`           | Compute and display row averages                |
| `-v`           | Verbose: print full matrix output                |
| `-g`           | Run **GPU-only** simulation                      |
| `--cpu-only`   | Run **CPU-only** simulation                      |
| `-t`           | Print **timing** information and **speedup**     |
