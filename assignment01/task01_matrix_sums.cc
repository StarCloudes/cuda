#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cstring>

// Function to get current time in microseconds
static inline double get_time_in_microseconds()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// rowSumAbs: Sums the absolute values of each row into a vector of length n.
void rowSumAbs(const float* matrix, float* rowSums, int n, int m)
{
    // Initialize row sums to zero
    for (int i = 0; i < n; i++) {
        rowSums[i] = 0.0f;
    }

    // Sum of absolute values for each row
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            rowSums[i] += std::fabs(matrix[i*m + j]);
        }
    }
}

// colSumAbs: Sums the absolute values of each column into a vector of length m.
void colSumAbs(const float* matrix, float* colSums, int n, int m)
{
    // Initialize column sums to zero
    for (int j = 0; j < m; j++) {
        colSums[j] = 0.0f;
    }

    // Sum of absolute values for each column
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            colSums[j] += std::fabs(matrix[i*m + j]);
        }
    }
}

// reduce: Sum up the components of a vector of length len.
float reduce(const float* vec, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += vec[i];
    }
    return sum;
}

int main(int argc, char* argv[])
{
    // Default values
    int n = 10;
    int m = 10;
    bool randomSeed = false;  // false = use fixed seed (1234567)
    bool printTiming = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            m = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r") == 0) {
            randomSeed = true;
        }
        else if (strcmp(argv[i], "-t") == 0) {
            printTiming = true;
        }
    }

    // Seed drand48
    if (randomSeed) {
        struct timeval myRandom;
        gettimeofday(&myRandom, NULL);
        srand48((long)myRandom.tv_usec);
    } else {
        srand48(1234567);
    }

    // Allocate matrix
    float* matrix = new float[n*m];

    // Fill matrix with random values in [-20, 20]
    for (int i = 0; i < n*m; i++) {
        matrix[i] = static_cast<float>(drand48() * 40.0 - 20.0);
    }

    // Allocate rowSums and colSums
    float* rowSums = new float[n];
    float* colSums = new float[m];

    double start, end;
    double rowSumTime = 0.0, colSumTime = 0.0;
    double rowReduceTime = 0.0, colReduceTime = 0.0;

    // 1. rowSumAbs
    start = get_time_in_microseconds();
    rowSumAbs(matrix, rowSums, n, m);
    end = get_time_in_microseconds();
    rowSumTime = end - start;

    // 2. colSumAbs
    start = get_time_in_microseconds();
    colSumAbs(matrix, colSums, n, m);
    end = get_time_in_microseconds();
    colSumTime = end - start;

    // 3. reduce row sums
    start = get_time_in_microseconds();
    float rowReduceValue = reduce(rowSums, n);
    end = get_time_in_microseconds();
    rowReduceTime = end - start;

    // 4. reduce column sums
    start = get_time_in_microseconds();
    float colReduceValue = reduce(colSums, m);
    end = get_time_in_microseconds();
    colReduceTime = end - start;

    // Print results
    std::cout << "Matrix size: " << n << " x " << m << std::endl;
    std::cout << "Sum of row absolute values (reduced)   = " << rowReduceValue << std::endl;
    std::cout << "Sum of column absolute values (reduced) = " << colReduceValue << std::endl;

    // Optional: Print timings
    if (printTiming) {
        std::cout << std::endl;
        std::cout << "Timing (microseconds):" << std::endl;
        std::cout << "  rowSumAbs         : " << rowSumTime << std::endl;
        std::cout << "  colSumAbs         : " << colSumTime << std::endl;
        std::cout << "  row reduce        : " << rowReduceTime << std::endl;
        std::cout << "  column reduce     : " << colReduceTime << std::endl;
    }

    // Cleanup
    delete[] matrix;
    delete[] rowSums;
    delete[] colSums;

    return 0;
}