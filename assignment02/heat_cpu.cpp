// src/main.cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <iomanip>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [-n rows] [-m cols] [-p iters] [-a (compute averages)]\n";
}

int main(int argc, char** argv) {
    // default size and iterations
    int n = 32, m = 32, p = 10;
    bool do_avg = false;
    // verbose diagnostics
    bool verbose = true;

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            p = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-a") == 0) {
            do_avg = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    // Allocate two n×m matrices
    std::vector<float> matA(n * m), matB(n * m);

    // Initialize column 0 boundary
    for (int i = 0; i < n; ++i) {
        float val = 0.98f * float((i + 1) * (i + 1)) / float(n * n);
        matA[i * m + 0] = val;
        matB[i * m + 0] = val;
    }

    // Initialize other cells
    for (int i = 0; i < n; ++i) {
        float base = matA[i * m + 0];
        for (int j = 1; j < m; ++j) {
            float factor = float((m - j) * (m - j)) / float(m * m);
            matA[i * m + j] = base * factor;
            matB[i * m + j] = base * factor;
        }
    }

    // Heat-propagation iterations
    for (int iter = 0; iter < p; ++iter) {
        auto &prev = (iter % 2 == 0 ? matA : matB);
        auto &next = (iter % 2 == 0 ? matB : matA);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                // keep column 0 fixed
                if (j == 0) {
                    next[i * m + j] = prev[i * m + j];
                    continue;
                }
                // wrap-around neighbors
                int jm2 = (j - 2 + m) % m;
                int jm1 = (j - 1 + m) % m;
                int jp1 = (j + 1) % m;
                int jp2 = (j + 2) % m;

                float sum = 1.60f * prev[i * m + jm2]
                          + 1.55f * prev[i * m + jm1]
                          +       prev[i * m + j]
                          + 0.60f * prev[i * m + jp1]
                          + 0.25f * prev[i * m + jp2];
                next[i * m + j] = sum / 5.0f;
            }
        }

        // Diagnostic: per-iteration total heat
        if (verbose) {
            float iter_sum = 0.0f;
            for (int idx = 0; idx < n * m; ++idx) {
                // whichever buffer is 'next'
                iter_sum += next[idx];
            }
            std::cout << "Iter " << iter
                      << " sum = " << iter_sum << "\n";
        }
    }

    // Pick final result buffer
    auto &result = (p % 2 == 0 ? matA : matB);

    // Diagnostic: final sum, min, max
    if (verbose) {
        float total_sum = 0.0f, minv = result[0], maxv = result[0];
        for (int idx = 0; idx < n * m; ++idx) {
            float v = result[idx];
            total_sum += v;
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        std::cout << "Final sum = " << total_sum
                  << ", min = " << minv
                  << ", max = " << maxv << "\n";
    }

    // Compute and print row averages when needed
    if (do_avg) {
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < n; ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                row_sum += result[i * m + j];
            }
            std::cout << "Row " << i
                      << " average = " << (row_sum / float(m))
                      << "\n";
        }
    }

    std::cout << "Done: n=" << n
              << ", m=" << m
              << ", iterations=" << p
              << ", averages=" << (do_avg ? "yes" : "no")
              << "\n";
    return 0;
}