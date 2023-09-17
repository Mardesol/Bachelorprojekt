#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu" 
#include "..\..\Matrix\matrixFloats.cu"

const int M1Rows = 2000;
const int M1Cols = 2000;
const int M2Rows = 2000;
const int M2Cols = 2000;
const int M3Rows = M1Rows;
const int M3Cols = M1Cols;

// CUDA kernel to add two matrices sequentially
__global__ void matrixAdditionSequential(float* M1, float* M2, float* M3) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M1Cols; j++) {
            M3[M1Rows * i + j] = M1[M1Rows * i + j] + M2[M1Rows * i + j];
        }
    }
}

// CUDA kernel to add two matrices in parallel, utilizing thread-level parallelism
__global__ void matrixAdditionParallelV1(float* M1, float* M2, float* M3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M3Rows * M3Cols) {
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism
__global__ void matrixAdditionParallelV2(float* M1, float* M2, float* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M3Rows && col < M3Cols) {
        int index = row * M3Rows + col;
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism, as well as shared memory
__global__ void matrixAdditionSharedMemory(float* M1, float* M2, float* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedMemory1[16];
    __shared__ float sharedMemory2[16];

    int index = row * M3Rows + col;

    // Load data into shared memory
    sharedMemory1[threadIdx.x] = M1[index];
    sharedMemory2[threadIdx.x] = M2[index];

    __syncthreads();  // Ensure all threads have loaded data

    if (row < M3Rows && col < M3Cols) {
        M3[index] = sharedMemory1[threadIdx.x] + sharedMemory2[threadIdx.x];
    }
}

int main() {
    // Timer measures the time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Define variables
    MatrixFloat M1;
    MatrixFloat M2;
    MatrixFloat M3;

    // Create the matrix objects
    M1 = createMatrixFloat(M1Rows, M1Cols);
    M2 = createMatrixFloat(M2Rows, M2Cols);
    M3 = createMatrixFloat(M3Rows, M3Cols);

    // Populate the matrices
    populateWithOnesFloat(M1);
    populateWithOnesFloat(M2);
    //populateWithRandomFloats(M1);
    //populateWithRandomFloats(M2);

    // Stop the setup timer
    endTimer(timer, "setup");

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Allocate memory for matrices on the GPU
    float* device_M1, * device_M2, * device_M3;

    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(float));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(float));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(float));

    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(float), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)");

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Start the matrix addition timer
    beginTimer(timer);

    // Launch the CUDA kernel to perform matrix addition
    matrixAdditionSharedMemory << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);

    // Stop the matrix addition timer
    endTimer(timer, "matrix addition (GPU)");

    // Start the data transfer timer (GPU -> CPU / Device -> Host)
    beginTimer(timer);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop the data transfer timer (GPU -> CPU / Device -> Host)
    endTimer(timer, "data transfer (GPU -> CPU)");

    // Open a new file to write the result into
    FILE* outputFile = fopen("resultFloats.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < M3Rows; i++) {
        for (int j = 0; j < M3Cols; j++) {
            fprintf(outputFile, "%f ", M3.data[i * M3Rows + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the result file
    fclose(outputFile);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // End the program
    return 0;
}