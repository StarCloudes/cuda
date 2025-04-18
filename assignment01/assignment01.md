

## **Task 1 - CPU calculation (Only in the CPU)**

1. **Command-line options:**
•	 -n : set number of rows (default 10).
​•	-m : set number of columns (default 10).
​•	-r : use a “random” seed based on the current time instead of a fixed seed.
​•	-t : print timings for each of the three steps.

2. **Compile and run with Makefile**
* **Compile** all  tasks code  using `make` 
```shell
kuangg@cuda01:~/homework/assignment01$ make
g++ -O3 -Wall -std=c++11 -o task01_matrix_sums task01_matrix_sum.cpp
g++ -O3 -Wall -std=c++11 -I/usr/local/cuda/include -c task02_matrix_sum.cpp -o task02_matrix_sum.o
nvcc -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75 -c task02_matrix_sum_cuda.cu -o task02_matrix_sum_cuda.o
g++ task02_matrix_sum.o task02_matrix_sum_cuda.o -o task02_matrix_sums -L/usr/local/cuda/lib64 -lcudart
```

* **Run** `make run_task01`  to see the result 
```shell
kuangg@cuda01:~/homework/assignment01$ make run_task01
./task01_matrix_sums -n 1000 -m 1000 -t
Matrix size: 1000 x 1000
Row sum reduction: 9999848.000000
Column sum reduction: 9999847.000000

Row sum calculation time: 1.049 ms
Column sum calculation time: 8.337 ms
Row sum reduction time: 0.001 ms
Column sum reduction time: 0.001 ms
```

**Short Explanation of Performance Differences**
1. **Row Summation**:  In C/C++, rows are stored one after the other in memory. When you add up a row, you’re **going through numbers that sit right next to each other,** so the CPU cache can grab them quickly. This makes row summation faster.
2. **Column Summation**:   For columns, you have to **jump from one row to the next** because the numbers aren’t next to each other in memory. This means the cache can’t help as much, so column summation tends to be slower.



### **Task 2 - parallel implementation (In the CPU and GPU)**

1. **Command-Line Arguments**:
  ​•	-n : number of rows (default = 10)
  ​•	-m : number of columns (default = 10)
  ​•	-r : use a random seed (current microseconds) instead of a fixed seed
  •	-b: threads per block(default 256)
  •	-t : show  timings

2. **Run** `make run_task02` 
```shell
kuangg@cuda01:~/homework/assignment01$ make run_task02
rm -f results.txt
/usr/local/cuda-12.6/bin/compute-sanitizer ./task02_matrix_sums -n 1000 -m 1000 -t
========= COMPUTE-SANITIZER
Performing CPU calculations...
Performing GPU calculations...

Matrix size: 1000 x 1000
Threads per block: 256

CPU Row sum reduction: 9999848.000000
GPU Row sum reduction: 9999853.000000
Relative error for Row sum reduction: 0.000000500 (0.000050001%)

CPU Column sum reduction: 9999847.000000
GPU Column sum reduction: 9999852.000000
Relative error for Column sum reduction: 0.000000500 (0.000050001%)

Timing Information:
Row sum calculation time - CPU: 0.892 ms, GPU: 4.948 ms, Speedup: 0.18x
Column sum calculation time - CPU: 7.155 ms, GPU: 3.036 ms, Speedup: 2.36x
Row sum reduction time - CPU: 0.001 ms, GPU: 0.225 ms, Speedup: 0.00x
Column sum reduction time - CPU: 0.001 ms, GPU: 0.126 ms, Speedup: 0.01x
Results written successfully to 'results.txt'
========= ERROR SUMMARY: 0 errors
```

**Explanation of Performance Differences**
1. **Result Accuracy:**

​	•	The CPU and GPU results for both row and column sum reductions are almost identical.
​	•	For example, the CPU row sum reduction is 9999848.000000 and the GPU is 9999853.000000. The relative error is extremely low (about 0.00005%), which shows that the GPU calculations are accurate.

2. **Timing Comparison:**

​	•	**Row Sum Calculation:**
​	•	CPU took 0.892 ms while the GPU took 4.948 ms.
​	•	The speedup is 0.18x, meaning the GPU was slower for this part, likely due to overheads associated with launching GPU kernels for a relatively simple computation.

​	•	**Column Sum Calculation:**
​	•	CPU took 7.155 ms and the GPU took 3.036 ms.
​	•	The speedup is 2.36x, which indicates that for column calculations the GPU is faster.

​	•	**Reduction Operations:**
​	•	The reduction times on the CPU are nearly negligible (0.001 ms) compared to the GPU times (0.225 ms for row reduction and 0.126 ms for column reduction).
​	•	This suggests that for these very fast operations, the GPU overhead does not pay off.

3. **Overall Observations:**

​	•	The GPU provides a benefit in some cases (like the column sum calculation) but may be slower in others (row sums and reductions) due to kernel launch overhead and the simplicity of the CPU operation.
​	•	The accuracy of the GPU implementation is excellent, as the computed values closely match the CPU’s results.



## **Task 3 - performance improvement**
1. **Run**  `make run_task03`   to get the **`task03_results.txt `**  results file

```shell
kuangg@cuda01:~/homework/assign01$ make run_task03
```

2. **Run** `python3 task03_plot.py `  to draw the performance picture.（**Make sure you have python3, Pandas and Matplotlib installed** ）



**Performance Analysis**:

![Screenshot 2025-03-06 at 11.04.07 PM](/Users/neil/Library/Application Support/typora-user-images/Screenshot 2025-03-06 at 11.04.07 PM.png)

1. **Overall Speedups:**

​	•	**Column Sum** shows the highest speedup on the GPU. For larger matrices (e.g., 10,000 x 10,000 or 25,000 x 25,000), the GPU consistently outperforms the CPU because the heavier workload makes kernel launches more worthwhile.
​	•	**Row Sum** has lower speedups and sometimes dips below 1 (meaning the CPU is faster), likely due to less efficient memory access patterns on the GPU or higher kernel overhead relative to the computation.

​	2.	**Reductions:**

​	•	Both row and column reductions generally have speedups close to or below 1. This indicates that the CPU can handle these final summations very quickly, making the GPU overhead less beneficial for such a small amount of work.

​	3.	**Threads per Block:**

​	•	Around 256 or 512 threads per block often yields the best performance for sums. Smaller or larger block sizes can reduce performance due to underutilizing or overloading the GPU.

​	4.	**Matrix Size Impact:**

​	•	As the matrix size increases, GPU performance tends to improve relative to the CPU for column sums because the extra data better hides kernel launch overhead.
​	•	For row sums and reductions, even large matrices do not provide a significant advantage, suggesting that their access patterns or the small reduction workload are not well suited to the GPU in this basic implementation.

In summary, **column sum calculations benefit the most from GPU parallelization**, especially at larger matrix sizes and optimal thread configurations. Row sums and reductions often perform better on the CPU or show only small GPU speedups, primarily due to overhead and less favorable memory access patterns.



## Task 4 - double precision testing

Based on the result graphs from the single‑precision tests, the best overall performance was achieved with a block size of around **256 threads per block**. 

1. **Run**`make run_task04`  to get the results of double precision results `task04_results.txt`.

```
kuangg@cuda01:~/homework/assign01$ make run_task04
```

2. **Run** `python3 comp_results.py`  to draw the performance picture.（**Make sure you have python3, matplotlib and Pandas installed** ）.

**Performance Analysis**:

![task04_speedup_plot](/Users/neil/Documents/code/cuda/assignment01_new/plots/task04_speedup_plot.png)

![task04_error_plot](/Users/neil/Documents/code/cuda/assignment01_new/plots/task04_error_plot.png)After modifing  code to use double-precision instead of single-precision and ran the same tests at the optimal block size of 256 for matrix sizes 1000, 5000, 10000, and 25000

​	1.	**Speedup**

​	•	Double-precision computations are generally **slower** than single-precision because each double-precision operation processes 64-bit floating-point values instead of 32-bit.
​	•	The **row sum** operations achieve higher speedup than the **column sum** operations, both in single and double precision.
​	•	As the matrix size increases, the difference in speedup between single and double precision becomes more noticeable.

​	2.	**Error (Precision)**

​	•	Double precision yields **lower relative errors** compared to single precision, especially at larger matrix sizes (e.g., 25000).
​	•	This improvement in accuracy comes at the cost of reduced speedup compared to single precision.

Overall, **single precision is faster** but **less accurate**, while **double precision is more accurate** but **slightly slower**. This trade-off is typical when moving from single- to double-precision floating-point arithmetic.