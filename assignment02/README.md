# 🧊 Heat Simulation - Assignment 02
This project implements a 1D heat propagation simulation using both CPU and GPU (CUDA), with optional timing, verification, and row-averaging features.

## 📁 File Structure

**main.cu** (Main program entrypoint. Handles argument parsing, memory management, CPU and GPU orchestration.)

**heat_cpu.h/.cpp** (CPU-side heat propagation and row average functions.)

**heat_gpu.cuh/.cu** (GPU kernel declarations and implementations.)

**real.h** (USE_DOUBLE or USE_FLOAT.)


## 🚀 Build Instructions

To compile the project, run: 
```
make heat_sim
make heat_sim_dp
```

## 🧪 Usage
```
./heat_sim [options]
./heat_sim -n 4 -m 128 -p 5 --cpu-only
./heat_sim -n 256 -m 256 -p 5 -g

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
