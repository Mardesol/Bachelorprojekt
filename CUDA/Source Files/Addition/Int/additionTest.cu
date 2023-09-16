#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu" 
#include "..\..\Matrix\matrixInts.cu"

const int rows = 100;
const int cols = 100;

const int M1Rows = rows;
const int M2Rows = rows;
const int M3Rows = rows;

const int M3Cols = cols;
const int M1Cols = cols;
const int M2Cols = cols;

// CUDA kernel to add two matrices sequentially
__global__ void matrixAdditionSequential(int* M1, int* M2, int* M3) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M1Cols; j++) {
            M3[M1Rows * i + j] = M1[M1Rows * i + j] + M2[M1Rows * i + j];
        }
    }
}

// CUDA kernel to add two matrices in parallel, utilizing thread level parallelism
__global__ void matrixAdditionParallelV1(int* M1, int* M2, int* M3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M3Rows * M3Cols) {
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism
__global__ void matrixAdditionParallelV2(int* M1, int* M2, int* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M3Rows && col < M3Cols) {
        int index = row * M3Rows + col;
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism, as well as shared memory
__global__ void matrixAdditionSharedMemory(int* M1, int* M2, int* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedMemory1[16];
    __shared__ int sharedMemory2[16];

    int index = row * M3Rows + col;

    // Load data into shared memory
    sharedMemory1[threadIdx.x] = M1[index];
    sharedMemory2[threadIdx.x] = M2[index];

    __syncthreads();  // Ensure all threads have loaded data

    if (row < M3Rows && col < M3Cols) {
        M3[index] = sharedMemory1[threadIdx.x] + sharedMemory2[threadIdx.x];
    }
}

// Function to measure and record execution times to a file
void measureAndRecordExecutionTimes(
    const char* outputFileName,
    Timer timer,
    void (*kernel)(int*, int*, int*),
    int* M1, int* M2, int* M3,
    dim3 gridDim, dim3 blockDim
) {
    // Open a new file to write the result into
    FILE* outputFile = fopen(outputFileName, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return;
    }

    for (int i = 0; i < 100; i++) {
        // Measure execution time for MMV1Sequential
        beginTimer(timer);
        cudaDeviceSynchronize();
        kernel <<<gridDim, blockDim>>>(M1, M2, M3);
        cudaDeviceSynchronize();
        float time = endTimerReturnTime(timer);

        fprintf(outputFile, "%f ms\n", time);
    }

    // Close the output file
    fclose(outputFile);
}

int main() {
    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Define variables
    Matrix M1;
    Matrix M2;
    Matrix M3;

    // Create the matrix objects
    M1 = createMatrix(M1Rows, M1Cols);
    M2 = createMatrix(M2Rows, M2Cols);
    M3 = createMatrix(M3Rows, M3Cols);

    // Populate the matrices
    populateWithOnes(M1);
    populateWithOnes(M2);

    // Stop the setup timer
    endTimer(timer, "setup");

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;

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

    // Measure and record execution times
    measureAndRecordExecutionTimes("Test/MA1SequentialResults.txt", timer, matrixAdditionSequential,   device_M1, device_M2, device_M3, gridDim, blockDim);
    measureAndRecordExecutionTimes("Test/MA2ParallelV1.txt",        timer, matrixAdditionParallelV1,   device_M1, device_M2, device_M3, gridDim, blockDim);
    measureAndRecordExecutionTimes("Test/MA3ParallelV2.txt",        timer, matrixAdditionParallelV2,   device_M1, device_M2, device_M3, gridDim, blockDim);
    measureAndRecordExecutionTimes("Test/MASharedMemory.txt",       timer, matrixAdditionSharedMemory, device_M1, device_M2, device_M3, gridDim, blockDim);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // Exit program
    return 0;
}