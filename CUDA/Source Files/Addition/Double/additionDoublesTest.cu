#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu" 
#include "..\..\Matrix\matrixDoubles.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const int rows = 200;
const int cols = 200;

const int M1Rows = rows;
const int M2Rows = rows;
const int M3Rows = rows;

const int M3Cols = cols;
const int M1Cols = cols;
const int M2Cols = cols;

// CUDA kernel to add two matrices sequentially
__global__ void matrixAdditionSequential(double* M1, double* M2, double* M3) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M1Cols; j++) {
            M3[M1Rows * i + j] = M1[M1Rows * i + j] + M2[M1Rows * i + j];
        }
    }
}

// CUDA kernel to add two matrices in parallel, utilizing thread-level parallelism
__global__ void matrixAdditionParallelV1(double* M1, double* M2, double* M3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M3Rows * M3Cols) {
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block-level parallelism
__global__ void matrixAdditionParallelV2(double* M1, double* M2, double* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M3Rows && col < M3Cols) {
        int index = row * M3Rows + col;
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block-level parallelism, as well as shared memory
__global__ void matrixAdditionSharedMemory(double* M1, double* M2, double* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double sharedMemory1[16];
    __shared__ double sharedMemory2[16];

    int index = row * M3Rows + col;

    // Load data into shared memory
    sharedMemory1[threadIdx.x] = M1[index];
    sharedMemory2[threadIdx.x] = M2[index];

    __syncthreads();  // Ensure all threads have loaded data

    if (row < M3Rows && col < M3Cols) {
        M3[index] = sharedMemory1[threadIdx.x] + sharedMemory2[threadIdx.x];
    }
}

// Function to measure kernel execution time
float measureKernelExecutionTime(
    void (*kernel)(double*, double*, double*),
    double* M1, double* M2, double* M3,
    dim3 gridDim, dim3 blockDim
) {
    Timer timer = createTimer();
    beginTimer(timer);

    cudaDeviceSynchronize();
    kernel << <gridDim, blockDim >> > (M1, M2, M3);
    cudaDeviceSynchronize();

    return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
    float* executionTimes,
    void (*kernel)(double*, double*, double*),
    double* M1, double* M2, double* M3,
    dim3 gridDim, dim3 blockDim
) {
    for (int i = 0; i < 100; i++) {
        // Measure execution time for the kernel
        float time = measureKernelExecutionTime(kernel, M1, M2, M3, gridDim, blockDim);
        executionTimes[i] = time;
    }
}

int main() {
    if (!additionCheck(M1Rows, M1Cols, M2Rows, M2Cols)) {
        perror("Matrices must have the same size");
        return 1;
    }

    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Define variables
    MatrixD M1;
    MatrixD M2;
    MatrixD M3;

    // Create the matrix objects
    M1 = createMatrixD(M1Rows, M1Cols);
    M2 = createMatrixD(M2Rows, M2Cols);
    M3 = createMatrixD(M3Rows, M3Cols);

    // Populate the matrices
    populateWithOnesD(M1);
    populateWithOnesD(M2);

    // Stop the setup timer
    endTimer(timer, "setup");

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Allocate memory for matrices on the GPU
    double* device_M1, * device_M2, * device_M3;

    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(int));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(int));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(int), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)");

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
        blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
    }

    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Create an array to store execution times for each kernel
    float executionTimes[4][100]; // 4 kernels, 100 executions each

    // Measure and record execution times for all kernels
    measureExecutionTimes(executionTimes[0], matrixAdditionSequential,      device_M1, device_M2, device_M3, gridDim, blockDim);
    measureExecutionTimes(executionTimes[1], matrixAdditionParallelV1,      device_M1, device_M2, device_M3, gridDim, blockDim);
    measureExecutionTimes(executionTimes[2], matrixAdditionParallelV2,      device_M1, device_M2, device_M3, gridDim, blockDim);
    measureExecutionTimes(executionTimes[3], matrixAdditionSharedMemory,    device_M1, device_M2, device_M3, gridDim, blockDim);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // Open the output file for writing in append mode
    FILE* outputFile = fopen("Test/Doubles-ExecutionTimes.csv", "a");
    if (outputFile == NULL) {
        perror("Unable to open the output file");
        return 1;
    }

    // Write execution times to the output file in separate columns
    fprintf(outputFile, "Sequential,ParallelV1,ParallelV2,SharedMemory\n");
    for (int i = 0; i < 100; i++) {
        fprintf(outputFile, "%f,%f,%f,%f\n",
            executionTimes[0][i],
            executionTimes[1][i],
            executionTimes[2][i],
            executionTimes[3][i]);
    }

    // Close the output file
    fclose(outputFile);

    // Exit program
    return 0;
}