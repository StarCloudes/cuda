#include "heat_cpu.h"
#include <cmath>
#include <iostream>
#include "real.h"

/**
 * @brief Simulates heat propagation on the CPU using a 2D matrix.
 * 
 * @param A Input matrix (current state) as a 1D vector.
 * @param B Output matrix (next state) as a 1D vector.
 * @param n Number of rows in the matrix.
 * @param m Number of columns in the matrix.
 * @param p Number of iterations to simulate.
 */
void cpu_heat_propagation(std::vector<real_t>& A, std::vector<real_t>& B, int n, int m, int p) {
    for (int iter = 0; iter < p; ++iter) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                // keep column 0 fixed
                if (j == 0) {
                    B[i * m + j] = A[i * m + j];
                } else {
                    // wrap-around neighbors
                    int jm2 = (j - 2 + m) % m;
                    int jm1 = (j - 1 + m) % m;
                    int jp1 = (j + 1) % m;
                    int jp2 = (j + 2) % m;

                    // Compute the weighted sum using the 5-point stencil formula
                    real_t sum = 1.60f * A[i * m + jm2]
                              + 1.55f * A[i * m + jm1]
                              +        A[i * m + j]
                              + 0.60f * A[i * m + jp1]
                              + 0.25f * A[i * m + jp2];
                    B[i * m + j] = sum / 5.0f;
                }
            }
        }
       std::swap(A, B);       
    }
   // std::cout << "cpu_heat_propagation \n";
}

/**
 * @brief Computes the average value of each row in a 2D matrix on the CPU.
 * 
 * @param mat Input matrix as a 1D vector.
 * @param avg Output vector to store the row averages.
 * @param n Number of rows in the matrix.
 * @param m Number of columns in the matrix.
 */
void compute_cpu_row_averages(const std::vector<real_t>& mat, std::vector<real_t>& avg, int n, int m) {
    for (int i = 0; i < n; ++i) {
        real_t sum = 0.0f;
        for (int j = 0; j < m; ++j) {
            sum += mat[i * m + j];
        }
        avg[i] = sum / m;
    }
    real_t total_sum = 0.0f, min_val = mat[0], max_val = mat[0];
    for (size_t i = 0; i < mat.size(); ++i) {
        total_sum += mat[i];
        if (mat[i] < min_val) min_val = mat[i];
        if (mat[i] > max_val) max_val = mat[i];
    }
    std::cout << "CPU result: Final sum = " << total_sum
              << ", min = " << min_val
              << ", max = " << max_val << std::endl;
}