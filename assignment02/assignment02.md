# Cylindrical Radiator Finite Differences model

The goal of this assignment is to write a model for propagation of heat inside a cylindrical radiator. 



## **Task 1 - CPU calculation **

### 1. **Objective**

Implement the heat‑propagation model on the CPU in single precision, with configurable grid size (n×m), number of iterations (p), and optional row‑average output.

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

```
./heat_cpu
./heat_cpu -n 30 -m 50 -p 20
```

Diagnostic output:

```
Iter 0 sum = 183.342
Iter 1 sum = 189.559
Iter 2 sum = 195.601
Iter 3 sum = 201.527
Iter 4 sum = 207.371
Iter 5 sum = 213.153
Iter 6 sum = 218.881
Iter 7 sum = 224.563
Iter 8 sum = 230.202
Iter 9 sum = 235.799
Iter 10 sum = 241.354
Iter 11 sum = 246.868
Iter 12 sum = 252.338
Iter 13 sum = 257.764
Iter 14 sum = 263.144
Iter 15 sum = 268.476
Iter 16 sum = 273.758
Iter 17 sum = 278.988
Iter 18 sum = 284.164
Iter 19 sum = 289.283
Final sum = 289.283, min = 0.000181621, max = 0.98
Done: n=30, m=50, iterations=20, averages=no
```

### 5. **Observations**

- **Monotonic Increase in Heat Sum**

  The total heat across all cells increased steadily from 183.34 to 289.28 over 20 iterations. This suggests that heat is correctly propagating from the fixed left boundary into the rest of the grid.

- **Expected Maximum Temperature Reached**

  The maximum observed value at the end of the simulation was exactly 0.98, which matches the theoretical maximum imposed by the boundary condition on column 0. This confirms correct initialization and preservation of boundary values.

- **Low Minimum Value Indicates Gradient Formation**

  The final minimum value (~0.00018) shows that heat had not fully reached the farthest columns yet, as expected in directional diffusion with relatively few steps. This confirms realistic spatial behavior.





