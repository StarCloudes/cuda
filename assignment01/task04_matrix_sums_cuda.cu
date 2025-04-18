#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <iomanip>   // For higher precision output

#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Utility: get CPU time in microseconds
//------------------------------------------------------------------------------
static inline double get_time_in_microseconds()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

//------------------------------------------------------------------------------
// Utility: check for CUDA errors
//------------------------------------------------------------------------------
static inline void checkCudaError(const char* message)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

//------------------------------------------------------------------------------
// CPU reference implementations (double precision)
//------------------------------------------------------------------------------
void rowSumAbsCPU(const double* matrix, double* rowSums, int n, int m)
{
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        rowSums[i] = sum;
    }
}

void colSumAbsCPU(const double* matrix, double* colSums, int n, int m)
{
    for (int j = 0; j < m; j++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        colSums[j] = sum;
    }
}

double reduceCPU(const double* vec, int len)
{
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += vec[i];
    }
    return sum;
}

//------------------------------------------------------------------------------
// GPU Kernels (double precision)
//------------------------------------------------------------------------------

// Kernel for rowSumAbs: each thread processes one row
__global__ void kernel_rowSumAbs(const double* d_matrix, double* d_rowSums, int n, int m)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        double sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += fabs(d_matrix[row*m + j]);
        }
        d_rowSums[row] = sum;
    }
}

// Kernel for colSumAbs: each thread processes one column
__global__ void kernel_colSumAbs(const double* d_matrix, double* d_colSums, int n, int m)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < m) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += fabs(d_matrix[i*m + col]);
        }
        d_colSums[col] = sum;
    }
}

// Naive reduce kernel (no shared memory, single-thread)
__global__ void kernel_reduce(const double* d_vec, double* d_result, int len)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            sum += d_vec[i];
        }
        d_result[0] = sum;
    }
}

//------------------------------------------------------------------------------
// Function to run the computation for a single (n, m, blockSize) in double precision,
// printing and saving results to "task04_results.txt" if desired.
//------------------------------------------------------------------------------
void runSingleTest(int n, int m, int blockSize,
                   bool randomSeed,
                   bool printCpuTiming,
                   bool printGpuTiming,
                   bool printSpeedup,
                   bool saveToFile)
{
    //------------------------------------------------------------------------
    // 1) Prepare host data
    //------------------------------------------------------------------------
    if (!randomSeed) {
        srand48(1234567);
    } else {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        srand48((long)tv.tv_usec);
    }

    double* h_matrix = new double[n*m];
    for (int i = 0; i < n*m; i++) {
        h_matrix[i] = static_cast<double>(drand48() * 40.0 - 20.0);
    }

    double* h_rowSums_cpu = new double[n];
    double* h_colSums_cpu = new double[m];

    // CPU Timings (in microseconds)
    double start_cpu, end_cpu;
    double timeRowCPU   = 0.0;
    double timeColCPU   = 0.0;
    double timeRedCPU_row = 0.0;
    double timeRedCPU_col = 0.0;

    //------------------------------------------------------------------------
    // 2) CPU calculations
    //------------------------------------------------------------------------

    // Row summation
    start_cpu = get_time_in_microseconds();
    rowSumAbsCPU(h_matrix, h_rowSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    timeRowCPU = end_cpu - start_cpu;

    // Column summation
    start_cpu = get_time_in_microseconds();
    colSumAbsCPU(h_matrix, h_colSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    timeColCPU = end_cpu - start_cpu;

    // Reduce row sums
    start_cpu = get_time_in_microseconds();
    double rowReduced_cpu = reduceCPU(h_rowSums_cpu, n);
    end_cpu = get_time_in_microseconds();
    timeRedCPU_row = end_cpu - start_cpu;

    // Reduce column sums
    start_cpu = get_time_in_microseconds();
    double colReduced_cpu = reduceCPU(h_colSums_cpu, m);
    end_cpu = get_time_in_microseconds();
    timeRedCPU_col = end_cpu - start_cpu;

    //------------------------------------------------------------------------
    // 3) GPU calculations
    //------------------------------------------------------------------------
    double *d_matrix, *d_rowSums, *d_colSums, *d_reduce;
    cudaMalloc((void**)&d_matrix,   n*m*sizeof(double));
    cudaMalloc((void**)&d_rowSums,  n*sizeof(double));
    cudaMalloc((void**)&d_colSums,  m*sizeof(double));
    cudaMalloc((void**)&d_reduce,   sizeof(double));
    checkCudaError("cudaMalloc");

    cudaMemcpy(d_matrix, h_matrix, n*m*sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError("memcpy H->D (matrix)");

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    double timeRowGPU = 0.0;
    double timeColGPU = 0.0;
    double timeRedGPU_row = 0.0;
    double timeRedGPU_col = 0.0;
    float tmpTime = 0.0f;

    int gridSizeRows = (n + blockSize - 1) / blockSize;
    int gridSizeCols = (m + blockSize - 1) / blockSize;

    // Row summation (GPU)
    cudaEventRecord(startEvent);
    kernel_rowSumAbs<<<gridSizeRows, blockSize>>>(d_matrix, d_rowSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_rowSumAbs");
    cudaEventElapsedTime(&tmpTime, startEvent, endEvent);
    timeRowGPU = tmpTime;  // CUDA events report time in milliseconds

    // Column summation (GPU)
    cudaEventRecord(startEvent);
    kernel_colSumAbs<<<gridSizeCols, blockSize>>>(d_matrix, d_colSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_colSumAbs");
    cudaEventElapsedTime(&tmpTime, startEvent, endEvent);
    timeColGPU = tmpTime;

    // Reduce row sums (GPU)
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_rowSums, d_reduce, n);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("reduce rowSums");
    double rowReduced_gpu = 0.0;
    cudaMemcpy(&rowReduced_gpu, d_reduce, sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&tmpTime, startEvent, endEvent);
    timeRedGPU_row = tmpTime;

    // Reduce column sums (GPU)
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_colSums, d_reduce, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("reduce colSums");
    double colReduced_gpu = 0.0;
    cudaMemcpy(&colReduced_gpu, d_reduce, sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&tmpTime, startEvent, endEvent);
    timeRedGPU_col = tmpTime;

    //------------------------------------------------------------------------
    // 4) Compare results & print with high precision
    //------------------------------------------------------------------------
    std::cout << "------------------------------------------------\n";
    std::cout << "Matrix: " << n << "x" << m 
              << " | blockSize=" << blockSize << std::endl;
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "CPU row-reduced  = " << rowReduced_cpu
              << ", GPU row-reduced  = " << rowReduced_gpu << std::endl;
    std::cout << "CPU col-reduced  = " << colReduced_cpu
              << ", GPU col-reduced  = " << colReduced_gpu << std::endl;

    // Compute relative error
    double rowError = 0.0;
    double colError = 0.0;
    if (std::fabs(rowReduced_cpu) > 1e-12) {
        rowError = std::fabs(rowReduced_gpu - rowReduced_cpu) / std::fabs(rowReduced_cpu);
    }
    if (std::fabs(colReduced_cpu) > 1e-12) {
        colError = std::fabs(colReduced_gpu - colReduced_cpu) / std::fabs(colReduced_cpu);
    }

    std::cout << "Row reduce error: " << (rowError * 100.0) << "%, "
              << "Col reduce error: " << (colError * 100.0) << "%\n";

    // Print timings
    if (printCpuTiming) {
        std::cout << "[CPU Timings - microseconds]\n"
                  << "  Row Summation : " << timeRowCPU << "\n"
                  << "  Col Summation : " << timeColCPU << "\n"
                  << "  Row Reduce    : " << timeRedCPU_row << "\n"
                  << "  Col Reduce    : " << timeRedCPU_col << std::endl;
    }
    if (printGpuTiming) {
        std::cout << "[GPU Timings - milliseconds]\n"
                  << "  Row Summation : " << timeRowGPU << "\n"
                  << "  Col Summation : " << timeColGPU << "\n"
                  << "  Row Reduce    : " << timeRedGPU_row << "\n"
                  << "  Col Reduce    : " << timeRedGPU_col << std::endl;
    }

    double rowSpeedup = 0.0, colSpeedup = 0.0, redSpeedup = 0.0;
    if (printSpeedup) {
        double rowCpuMS = timeRowCPU / 1000.0;
        double colCpuMS = timeColCPU / 1000.0;
        double redCpuMS = (timeRedCPU_row + timeRedCPU_col) / 1000.0;

        if (timeRowGPU > 1e-6) rowSpeedup = rowCpuMS / timeRowGPU;
        if (timeColGPU > 1e-6) colSpeedup = colCpuMS / timeColGPU;
        if ((timeRedGPU_row + timeRedGPU_col) > 1e-6) redSpeedup = redCpuMS / (timeRedGPU_row + timeRedGPU_col);

        std::cout << "[Speedups: CPU/GPU]\n"
                  << "  Row Summation Speedup : " << rowSpeedup << "\n"
                  << "  Col Summation Speedup : " << colSpeedup << "\n"
                  << "  Reduce Speedup        : " << redSpeedup << std::endl;
    }
    std::cout << "------------------------------------------------\n\n";

    //------------------------------------------------------------------------
    // 5) Save results to file (if requested)
    //------------------------------------------------------------------------
    if (saveToFile) {
        std::ofstream ofs("task04_results.txt", std::ios::app);
        // CSV header format:
        // n,m,blockSize,rowReducedCPU,rowReducedGPU,colReducedCPU,colReducedGPU,
        // rowError,colError,timeRowCPU_us,timeColCPU_us,timeRowRedCPU_us,timeColRedCPU_us,
        // timeRowGPU_ms,timeColGPU_ms,timeRowRedGPU_ms,timeColRedGPU_ms,
        // rowSpeedup,colSpeedup,redSpeedup
        ofs << n << "," << m << "," << blockSize << ","
            << rowReduced_cpu << "," << rowReduced_gpu << ","
            << colReduced_cpu << "," << colReduced_gpu << ","
            << rowError << "," << colError << ","
            << timeRowCPU << "," << timeColCPU << "," << timeRedCPU_row << "," << timeRedCPU_col << ","
            << timeRowGPU << "," << timeColGPU << "," << timeRedGPU_row << "," << timeRedGPU_col << ","
            << rowSpeedup << "," << colSpeedup << "," << redSpeedup << "\n";
        ofs.close();
    }

    // Cleanup
    delete[] h_matrix;
    delete[] h_rowSums_cpu;
    delete[] h_colSums_cpu;

    cudaFree(d_matrix);
    cudaFree(d_rowSums);
    cudaFree(d_colSums);
    cudaFree(d_reduce);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);
}

//------------------------------------------------------------------------------
// Main: run single test OR run an automated test loop
//------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Defaults for single test
    int n = 10;
    int m = 10;
    int blockSize = 256;
    bool randomSeed   = false;
    bool printCpuTime = false;
    bool printGpuTime = false;
    bool printSpeedup = false;
    bool autoTest     = false;  // flag for automated test loop
    bool saveToFile   = false;  // flag to save results in task04_results.txt

    // Parse arguments, e.g., -n 1000 -m 1000 -bs 256 -t -gt -sp -auto -save
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            n = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
            m = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-bs") == 0 && i+1 < argc) {
            blockSize = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-r") == 0) {
            randomSeed = true;
        }
        else if (strcmp(argv[i], "-t") == 0) {
            printCpuTime = true;
        }
        else if (strcmp(argv[i], "-gt") == 0) {
            printGpuTime = true;
        }
        else if (strcmp(argv[i], "-sp") == 0) {
            printSpeedup = true;
        }
        else if (strcmp(argv[i], "-auto") == 0) {
            autoTest = true;
        }
        else if (strcmp(argv[i], "-save") == 0) {
            saveToFile = true;
        }
    }

    // Overwrite task04_results.txt with header if saveToFile is enabled
    if (saveToFile) {
        std::ofstream ofs("task04_results.txt", std::ios::out);
        ofs << "n,m,blockSize,"
            << "rowReducedCPU,rowReducedGPU,colReducedCPU,colReducedGPU,"
            << "rowError,colError,"
            << "timeRowCPU_us,timeColCPU_us,timeRowRedCPU_us,timeColRedCPU_us,"
            << "timeRowGPU_ms,timeColGPU_ms,timeRowRedGPU_ms,timeColRedGPU_ms,"
            << "rowSpeedup,colSpeedup,redSpeedup\n";
        ofs.close();
    }

    // If not doing autoTest, run one test with user-provided parameters
    if (!autoTest) {
        runSingleTest(n, m, blockSize, randomSeed,
                      printCpuTime, printGpuTime, printSpeedup, saveToFile);
        return 0;
    }

    // Otherwise, run the automated test loop:
    const int sizes[]   = {1000, 5000, 10000, 25000};
    const int bsizes[]  = {4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for (int s = 0; s < 4; s++)
    {
        for (int b = 0; b < 9; b++)
        {
            runSingleTest(sizes[s], sizes[s], bsizes[b], randomSeed,
                          printCpuTime, printGpuTime, printSpeedup, saveToFile);
        }
    }

    return 0;
}