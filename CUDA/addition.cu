#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.cu"
#include "timer.cu"


// CUDA kernel to add two matrices
__global__ void matrixAdditionSimple(int* M1, int* M2, int* M3, int M3R, int M3C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M3R * M3C) {
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel
__global__ void matrixAddition(int* M1, int* M2, int* M3, int M3R, int M3C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M3R && col < M3C) {
        int index = row * M3R + col;
        M3[index] = M1[index] + M2[index];
    }
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
    int M1Rows = 5;
    int M1Cols = 5;
    int M2Rows = 5;
    int M2Cols = 5;
    int M3Rows = M1Rows;
    int M3Cols = M1Cols;

    // Create the matrix objects
    M1 = createMatrix(M1Rows, M1Cols);
    M2 = createMatrix(M2Rows, M2Cols);
    M3 = createMatrix(M3Rows, M3Cols);

    // Populate the matrices
    //populateWithOnes(M1);
    //populateWithOnes(M2);
    populateWithRandomInts(M1);
    populateWithRandomInts(M2);


    // Stop the setup timer
    endTimer(timer, "setup");
    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;

    // Allocate memory for matrices on the GPU
    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(int));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(int));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(int), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)");

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Start the matrix addition timer
    beginTimer(timer);

    // Launch the CUDA kernel to perform matrix addition
    matrixAdditionSimple <<<gridDim, blockDim>>> (device_M1, device_M2, device_M3, M3Rows, M3Cols);

    // Stop the matrix addition timer
    endTimer(timer, "matrix addition (GPU)");

    // Start the data transfer timer (GPU -> CPU / Device -> Host)
    beginTimer(timer);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop the  data transfer timer (GPU -> CPU / Device -> Host)
    endTimer(timer, "data transfer (GPU -> CPU)");

    // Open a new file to write the result into
    FILE* outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < M3Rows; i++) {
        for (int j = 0; j < M3Cols; j++) {
            fprintf(outputFile, "%d ", M3.data[i * M3Rows + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the result file
    fclose(outputFile);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // End program
    return 0;
}
