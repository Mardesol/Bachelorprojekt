#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu" 
#include "..\..\Matrix\matrixDoubles.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
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

    //Setup a CPU comparison matrix
    MatrixD MCPU = createMatrixD(M3Rows, M3Cols);
    additionDouble(M1.data, M2.data, MCPU.data, M3Rows, M3Cols);

    // Stop the setup timer
    endTimer(timer, "setup");

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Create the matrix objects to be stored on the device
    double* device_M1, * device_M2, * device_M3;  

    // Allocate memory for matrices on the GPU
    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(double));  
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(double));  
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(double));  

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(double), cudaMemcpyHostToDevice);  

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)");

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Start the matrix addition timer
    beginTimer(timer);

    // Launch the CUDA kernel to perform matrix addition
    matrixAdditionSequential <<<gridDim, blockDim >>> (device_M1, device_M2, device_M3);

    // Stop the matrix addition timer
    endTimer(timer, "matrix addition (GPU)");

    // Start the data transfer timer (GPU -> CPU / Device -> Host)
    beginTimer(timer);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(double), cudaMemcpyDeviceToHost);  

    // Stop the data transfer timer (GPU -> CPU / Device -> Host)
    endTimer(timer, "data transfer (GPU -> CPU)");

    // Open a new file to write the result into
    FILE* outputFile = fopen("resultDoubles.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < M3Rows; i++) {
        for (int j = 0; j < M3Cols; j++) {
            fprintf(outputFile, "%lf ", M3.data[i * M3Rows + j]);  // Change format specifier to %lf for double
        }
        fprintf(outputFile, "\n");
    }

    // Close the result file
    fclose(outputFile);

    //Validate result by comparing to CPU calculations
    bool valid = compareMatricesDouble(MCPU.data, M3.data, M3Rows, M3Cols);
    if (valid) {
        printf("Matrix addition results match!\n");
    }
    else {
        printf("Matrix addition results do not match.\n");
        // Write the matrices to text files for analysis
        FILE* outputFile1 = fopen("resultDoublesCPU.txt", "w");
        if (outputFile1 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile1, "%lf ", MCPU.data[i * M3Rows + j]);  // Change format specifier to %lf for double
            }
            fprintf(outputFile1, "\n");
        }

        // Close the result file
        fclose(outputFile1);

        FILE* outputFile2 = fopen("resultDoublesGPU.txt", "w");
        if (outputFile2 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile2, "%lf ", M3.data[i * M3Rows + j]);  // Change format specifier to %lf for double
            }
            fprintf(outputFile2, "\n");
        }

        // Close the result file
        fclose(outputFile2);
    }

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // Exit program
    return 0;
}
