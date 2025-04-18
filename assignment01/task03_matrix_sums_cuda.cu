#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cstring>
#include <iomanip>   // Added for higher precision output

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
// CPU reference implementations
//------------------------------------------------------------------------------
void rowSumAbsCPU(const float* matrix, float* rowSums, int n, int m)
{
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        rowSums[i] = sum;
    }
}

void colSumAbsCPU(const float* matrix, float* colSums, int n, int m)
{
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += std::fabs(matrix[i*m + j]);
        }
        colSums[j] = sum;
    }
}

float reduceCPU(const float* vec, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += vec[i];
    }
    return sum;
}

//------------------------------------------------------------------------------
// GPU Kernels
//------------------------------------------------------------------------------

// Kernel for rowSumAbs: each thread processes one row
__global__ void kernel_rowSumAbs(const float* d_matrix, float* d_rowSums, int n, int m)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            sum += fabsf(d_matrix[row*m + j]);
        }
        d_rowSums[row] = sum;
    }
}

// Kernel for colSumAbs: each thread processes one column
__global__ void kernel_colSumAbs(const float* d_matrix, float* d_colSums, int n, int m)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < m) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += fabsf(d_matrix[i*m + col]);
        }
        d_colSums[col] = sum;
    }
}

// Naive reduce kernel (no shared memory, single-thread)
__global__ void kernel_reduce(const float* d_vec, float* d_result, int len)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            sum += d_vec[i];
        }
        d_result[0] = sum;
    }
}

//------------------------------------------------------------------------------
// A function that runs the entire computation for a single (n, m, blockSize)
// and stores the data to "task03_results.txt".
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

    float* h_matrix = new float[n*m];
    for (int i = 0; i < n*m; i++) {
        h_matrix[i] = static_cast<float>(drand48() * 40.0 - 20.0);
    }

    float* h_rowSums_cpu = new float[n];
    float* h_colSums_cpu = new float[m];

    // CPU Timings
    double start_cpu, end_cpu;
    double timeRowCPU   = 0.0;
    double timeColCPU   = 0.0;
    double timeRedCPU_row = 0.0;
    double timeRedCPU_col = 0.0;

    //------------------------------------------------------------------------
    // 2) CPU calculations
    //------------------------------------------------------------------------

    // row sums
    start_cpu = get_time_in_microseconds();
    rowSumAbsCPU(h_matrix, h_rowSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    timeRowCPU = end_cpu - start_cpu;

    // column sums
    start_cpu = get_time_in_microseconds();
    colSumAbsCPU(h_matrix, h_colSums_cpu, n, m);
    end_cpu = get_time_in_microseconds();
    timeColCPU = end_cpu - start_cpu;

    // reduce row sums
    start_cpu = get_time_in_microseconds();
    float rowReduced_cpu = reduceCPU(h_rowSums_cpu, n);
    end_cpu = get_time_in_microseconds();
    timeRedCPU_row = end_cpu - start_cpu;

    // reduce col sums
    start_cpu = get_time_in_microseconds();
    float colReduced_cpu = reduceCPU(h_colSums_cpu, m);
    end_cpu = get_time_in_microseconds();
    timeRedCPU_col = end_cpu - start_cpu;

    //------------------------------------------------------------------------
    // 3) GPU calculations
    //------------------------------------------------------------------------
    float *d_matrix, *d_rowSums, *d_colSums, *d_reduce;
    cudaMalloc((void**)&d_matrix,   n*m*sizeof(float));
    cudaMalloc((void**)&d_rowSums,  n*sizeof(float));
    cudaMalloc((void**)&d_colSums,  m*sizeof(float));
    cudaMalloc((void**)&d_reduce,   sizeof(float));
    checkCudaError("cudaMalloc");

    cudaMemcpy(d_matrix, h_matrix, n*m*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("memcpy H->D (matrix)");

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    float timeRowGPU = 0.0f;
    float timeColGPU = 0.0f;
    float timeRedGPU_row = 0.0f;
    float timeRedGPU_col = 0.0f;
    float tmpTime    = 0.0f;

    int gridSizeRows = (n + blockSize - 1) / blockSize;
    int gridSizeCols = (m + blockSize - 1) / blockSize;

    // row summation (GPU)
    cudaEventRecord(startEvent);
    kernel_rowSumAbs<<<gridSizeRows, blockSize>>>(d_matrix, d_rowSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_rowSumAbs");
    cudaEventElapsedTime(&timeRowGPU, startEvent, endEvent);

    // column summation (GPU)
    cudaEventRecord(startEvent);
    kernel_colSumAbs<<<gridSizeCols, blockSize>>>(d_matrix, d_colSums, n, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("kernel_colSumAbs");
    cudaEventElapsedTime(&timeColGPU, startEvent, endEvent);

    // reduce row sums (GPU)
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_rowSums, d_reduce, n);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("reduce rowSums");
    float rowReduced_gpu = 0.0f;
    cudaMemcpy(&rowReduced_gpu, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&tmpTime, startEvent, endEvent);
    timeRedGPU_row = tmpTime;

    // reduce col sums (GPU)
    cudaEventRecord(startEvent);
    kernel_reduce<<<1,1>>>(d_colSums, d_reduce, m);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    checkCudaError("reduce colSums");
    float colReduced_gpu = 0.0f;
    cudaMemcpy(&colReduced_gpu, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);
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
    float rowError = 0.0f;
    float colError = 0.0f;
    if (fabsf(rowReduced_cpu) > 1e-6f) {
        rowError = fabsf(rowReduced_gpu - rowReduced_cpu) / fabsf(rowReduced_cpu);
    }
    if (fabsf(colReduced_cpu) > 1e-6f) {
        colError = fabsf(colReduced_gpu - colReduced_cpu) / fabsf(colReduced_cpu);
    }

    std::cout << "Row reduce error: " << (rowError * 100.0f) << "%, "
              << "Col reduce error: " << (colError * 100.0f) << "%\n";

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

    // Speedup
    double rowSpeedup = 0.0, colSpeedup = 0.0, redSpeedup_row = 0.0, redSpeedup_col = 0.0;
    if (printSpeedup) {
        double rowCpuMS = timeRowCPU / 1000.0;
        double colCpuMS = timeColCPU / 1000.0;
        double redCpuMS_row = timeRedCPU_row / 1000.0;
        double redCpuMS_col = timeRedCPU_col / 1000.0;

        if (timeRowGPU > 1e-6) rowSpeedup = rowCpuMS / timeRowGPU;
        if (timeColGPU > 1e-6) colSpeedup = colCpuMS / timeColGPU;
        if (timeRedGPU_row > 1e-6) redSpeedup_row = redCpuMS_row / timeRedGPU_row;
        if (timeRedGPU_col > 1e-6) redSpeedup_col = redCpuMS_col / timeRedGPU_col;

        std::cout << "[Speedups: CPU/GPU]\n"
                  << "  Row Summation Speedup : " << rowSpeedup << "\n"
                  << "  Col Summation Speedup : " << colSpeedup << "\n"
                  << "  Row Reduce Speedup    : " << redSpeedup_row << "\n"
                  << "  Col Reduce Speedup    : " << redSpeedup_col << std::endl;
    }
    std::cout << "------------------------------------------------\n\n";

    //------------------------------------------------------------------------
    // 5) Save results to file (if requested)
    //------------------------------------------------------------------------
    if (saveToFile) {
        std::ofstream ofs("task03_results.txt", std::ios::app);
        // CSV header format:
        // n,m,blockSize,rowReducedCPU,rowReducedGPU,colReducedCPU,colReducedGPU,
        // rowError,colError,timeRowCPU_us,timeColCPU_us,timeRowRedCPU_us,timeColRedCPU_us,
        // timeRowGPU_ms,timeColGPU_ms,timeRowRedGPU_ms,timeColRedGPU_ms,
        // rowSpeedup,colSpeedup,rowRedSpeedup,colRedSpeedup
        ofs << n << "," << m << "," << blockSize << ","
            << rowReduced_cpu << "," << rowReduced_gpu << ","
            << colReduced_cpu << "," << colReduced_gpu << ","
            << rowError << "," << colError << ","
            << timeRowCPU << "," << timeColCPU << ","
            << timeRedCPU_row << "," << timeRedCPU_col << ","
            << timeRowGPU << "," << timeColGPU << ","
            << timeRedGPU_row << "," << timeRedGPU_col << ","
            << rowSpeedup << "," << colSpeedup << ","
            << redSpeedup_row << "," << redSpeedup_col << "\n";
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
    bool autoTest     = false;  // new flag for automated test loop
    bool saveToFile   = false;  // new flag to save results in task03_results.txt

    // Parse arguments
    // e.g., -n 1000 -m 1000 -bs 256 -t -gt -sp -auto -save
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

    // Save results: OVERWRITE the file and write header once
    if (saveToFile) {
        std::ofstream ofs("task03_results.txt", std::ios::out);
        ofs << "n,m,blockSize,"
            << "rowReducedCPU,rowReducedGPU,colReducedCPU,colReducedGPU,"
            << "rowError,colError,"
            << "timeRowCPU_us,timeColCPU_us,timeRowRedCPU_us,timeColRedCPU_us,"
            << "timeRowGPU_ms,timeColGPU_ms,timeRowRedGPU_ms,timeColRedGPU_ms,"
            << "rowSpeedup,colSpeedup,rowRedSpeedup,colRedSpeedup\n";
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
            int testN  = sizes[s];
            int testM  = sizes[s];
            int testBS = bsizes[b];

            runSingleTest(testN, testM, testBS, randomSeed,
                          printCpuTime, printGpuTime, printSpeedup, saveToFile);
        }
    }

    return 0;
}