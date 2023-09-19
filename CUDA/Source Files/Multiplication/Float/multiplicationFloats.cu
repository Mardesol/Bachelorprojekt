#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Matrix\matrixFloats.cu" // Assuming you have a matrixFloats.cu file
#include "..\..\Timer\timer.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const int M1Rows = 200;
const int M1Cols = 200;
const int M2Rows = 200;
const int M2Cols = 200;

__global__ void MMV1Sequential(float* M1, float* M2, float* M3) {

    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M2Cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M1Cols; k++) {
                sum += M1[i * M1Cols + k] *
                    M2[k * M2Cols + j];
            }
            M3[i * M2Cols + j] = sum;
        }
    }
}

__global__ void MMV2Parallelism(float* M1, float* M2, float* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M1Rows && col < M2Cols) {
        float sum = 0.0f;

        for (int i = 0; i < M1Cols; i++) {
            sum += M1[row * M1Cols + i] *
                M2[i * M2Cols + col];
        }
        M3[row * M2Cols + col] = sum;
    }
}

__global__ void MMV3SharedMemoryAndTiling(float* M1, float* M2, float* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory
    __shared__ float sharedMemory1[16];
    __shared__ float sharedMemory2[16];

    float sum = 0.0f;

    // Read into shared memory in a coalescing manner
    for (int i = 0; i < M1Cols; i += blockDim.x) {
        sharedMemory1[threadIdx.x] = (row < M1Rows && i + threadIdx.x < M1Cols) ? M1[row * M1Cols + i + threadIdx.x] : 0.0f;
        sharedMemory2[threadIdx.y] = (i + threadIdx.y < M1Cols && col < M2Cols) ? M2[(i + threadIdx.y) * M2Cols + col] : 0.0f;

        __syncthreads();

        // Perform the multiplication
        for (int j = 0; j < blockDim.x; j++) {
            sum += sharedMemory1[j] * sharedMemory2[j];
        }

        __syncthreads();
    }

    if (row < M1Rows && col < M2Cols) {
        M3[row * M2Cols + col] = sum;
    }
}

int main() {
    if (!multiplicationCheck(M1Cols, M2Rows)) {
        perror("Matrices must be compatible");
        return 1;
    }

    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Define variables
    MatrixF M1;
    MatrixF M2;
    MatrixF M3;
    int M3Rows = M1Rows;
    int M3Cols = M2Cols;

    // Create the matrix objects
    M1 = createMatrixF(M1Rows, M1Cols);
    M2 = createMatrixF(M2Rows, M2Cols);
    M3 = createMatrixF(M3Rows, M3Cols);

    // Populate the matrices
    populateWithOnesF(M1);
    populateWithOnesF(M2);

    //Setup a CPU comparison matrix
    MatrixF MCPU = createMatrixF(M3Rows, M3Cols);
    additionFloat(M1.data, M2.data, MCPU.data, M3Rows, M3Cols);

    // Stop the setup timer
    endTimer(timer, "setup");

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Create the matrix objects to be stored on the device
    float* device_M1, * device_M2, * device_M3;

    // Allocate memory for matrices on the GPU
    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(float));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(float));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(float));

    // Copy data from host to device
    // The data is matrix 1 and 2
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(float), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)");

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
        blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
    }

    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Start the matrix addition timer
    beginTimer(timer);

    // Launch the CUDA kernel to perform matrix multiplication
    cudaDeviceSynchronize();
    MMV1Sequential << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);
    cudaDeviceSynchronize();

    // Stop the matrix multiplication timer
    endTimer(timer, "matrix multiplication (GPU)");

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

    //Validate result by comparing to CPU calculations
    bool valid = compareMatricesFloat(MCPU.data, M3.data, M3Rows, M3Cols);
    if (valid) {
        printf("Matrix multiplication results match!\n");
    }
    else {
        printf("Matrix multiplication results do not match.\n");
        // Write the matrices to text files for analysis
        FILE* outputFile1 = fopen("resultFloatsCPU.txt", "w");
        if (outputFile1 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile1, "%f ", MCPU.data[i * M3Rows + j]);  // Change format specifier to %lf for double
            }
            fprintf(outputFile1, "\n");
        }

        // Close the result file
        fclose(outputFile1);

        FILE* outputFile2 = fopen("resultFloatsGPU.txt", "w");
        if (outputFile2 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile2, "%f ", M3.data[i * M3Rows + j]);  // Change format specifier to %lf for double
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