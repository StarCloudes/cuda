# Cylindrical Radiator Finite Differences model

The goal of this assignment is to write a model for propagation of heat inside a cylindrical radiator. 



## **Task 1 - CPU calculation **

### 1. **Objective** and code 

Implement the heat‑propagation model on the CPU in single precision, with configurable grid size (n×m), number of iterations (p), and optional row‑average output.

| **File**     | **Role**           | Purpose                                                      |
| ------------ | ------------------ | ------------------------------------------------------------ |
| heat_cpu.cpp | CPU Implementation | Implements CPU-side heat propagation and row averaging logic. |
| heat_cpu.h   | CPU Declaration    | Declares reusable CPU function interfaces for main.cu.       |

### 2. **Code Structure**

- **heat_up.cpp** contains:

  1. **CLI parsing** (-n, -m, -p, -a)

  2. **Matrix allocation**: two std::vector<float> of size n*m

  3. **Initialization**

     - **Boundary** (column 0):
       $$
       T[i][0] = 0.98 \times \frac{(i+1)^2}{n^2}
       $$

     - **Interior**:
       $$
       T[i][j] = T[i][0]\times\frac{(m-j)^2}{m^2},\quad j\ge1
       $$

  4. **Heat‑propagation loop** (iterating p steps)

     - Alternates between two buffers (matA ↔ matB)

     - For each cell j>0, applies directional five‑point stencil with wrap‑around:
       $$
       T_{\rm new}[j] = \frac{1.60\,T_{\rm old}[j-2] + 1.55\,T_{\rm old}[j-1] + 1.00\,T_{\rm old}[j] + 0.60\,T_{\rm old}[j+1] + 0.25\,T_{\rm old}[j+2]}{5.0}
       $$

     - Indices use $(j±k + m)%m$, so that positions $m–1, m–2$ correctly wrap to columns 0,1.

  5. **Diagnostics** :

     - Per‑iteration total sum

     - Final sum, minimum and maximum

  6. **Row‑average** (-a): computes and prints the average of each row at the end.

  

### 3. **Correctness & Edge‑Cases**

- **Circular wrap‑around**: modulo indexing ensures backward propagation from column 0→m–2 and 1→m–1.
- **Non‑square grids**: loops and modulo always use the actual m value. No assumption n==m.
- **Boundary fixed**: column 0 is copied each iteration, never updated.



### 4. **Usage Example**

In this task, we implemented **directional horizontal heat propagation** entirely on the CPU using finite differences. The radiator model includes cyclic wrap-around at each row to simulate a cylindrical pipe system. The computation was executed using the following parameters:

```
./heat_sim --cpu-only 
./heat_sim -n 4 -m 128 -p 5 --cpu-only
./heat_sim -n 4 -m 128 -p 5 --cpu-only -a 
./heat_sim -n 4 -m 128 -p 5 --cpu-only -a -v
```

Module output:![Screenshot 2025-04-19 at 2.13.45 AM](/Users/neil/Library/Application Support/typora-user-images/Screenshot 2025-04-19 at 2.13.45 AM.png)

![Screenshot 2025-04-19 at 2.12.43 AM](/Users/neil/Library/Application Support/typora-user-images/Screenshot 2025-04-19 at 2.12.43 AM.png)

### 5. **Observations**

- After 5 iterations, the **final matrix values** show clear and smooth horizontal heat diffusion patterns along each row. Heat values increase from left to right, following the direction of flow and weighted stencil propagation.
- The **boundary condition** (leftmost column) remains stable and precomputed as expected. It influences the rest of each row’s values based on the stencil weights.
- The **final total sum** was above, which confirms correct propagation within the valid float range.
- The **row averages** also increased gradually from top to bottom. This is consistent with the quadratic boundary condition, where lower rows have higher base temperatures and thus higher propagated values.



## Task 2  GPU **Implementation**

In this section, we extend the CPU-based heat propagation model from Task 1 by implementing its GPU counterpart using CUDA. The goal is to accelerate the computation of directional heat propagation across a 2D cylindrical matrix, using parallel threads and efficient memory access.

### 1. **Implementation Highlights**

| **File**     | **Role**       | **Purpose**                               |
| ------------ | -------------- | ----------------------------------------- |
| main.cu      | Orchestration  | Decouples logic; handles input/output     |
| heat_gpu.cuh | Declaration    | Clean and modular; prevents circular deps |
| heat_gpu.cu  | Implementation | Encapsulates CUDA logic; easy to maintain |

- **CUDA Kernel: Heat Propagate kernel**

  This kernel ensures wrap-around indexing to simulate the cylindrical structure.

-  **Iterative Propagation: launch_heat_propagation**：
  - Configure the grid and block dimensions
  - Call the kernel repeatedly for a given number of iterations
  - Alternate between two device buffers (d_A, d_B) to avoid overwriting data during propagation
- **CUDA Kernel: Row Average Computation**

​	Each block handles one row; each thread handles multiple elements (columns).

### 2. **Usage Example**

```
./heat_sim -n 256 -m 256 -p 5 -g
./heat_sim -n 256 -m 256 -p 5 -v
./heat_sim -n 256 -m 256 -p 5 -a 
./heat_sim -n 256 -m 256 -p 5 -t 
```

![Screenshot 2025-04-19 at 2.30.44 AM](/Users/neil/Library/Application Support/typora-user-images/Screenshot 2025-04-19 at 2.30.44 AM.png)

### **3.Observations**

- The CUDA implementation of the heat propagation simulation was successfully completed. As shown in the timing results, the GPU achieved a **speedup of over 73x** compared to the CPU for kernel + average computations. This demonstrates a significant performance benefit from GPU parallelism, especially on large grids (e.g., 256×256).
- In terms of correctness, the final GPU matrix perfectly matched the CPU result with a **maximum matrix difference of 0.000000**, ensuring the implementation is both fast and accurate.