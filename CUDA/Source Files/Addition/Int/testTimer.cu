#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu"
#include "..\..\Matrix\matrixInts.cu"

const int M1Rows = 5000;
const int M1Cols = 5000;
const int M2Rows = 5000;
const int M2Cols = 5000;
const int M3Rows = M1Rows;
const int M3Cols = M1Cols;



// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism, as well as shared memory
__global__ void matrixAdditionSharedMemory1D(int* M1, int* M2, int* M3, float* elapsedTime) {
    __shared__ int sharedMemory1[16];
    __shared__ int sharedMemory2[16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = row * M3Cols + col;

    // Load data into shared memory
    int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;
    sharedMemory1[sharedIdx] = M1[index];
    sharedMemory2[sharedIdx] = M2[index];

    __syncthreads();  // Ensure all threads have loaded data

    if (row < M3Rows && col < M3Cols) {
        // Start the timer
        clock_t start = clock();

        M3[index] = sharedMemory1[sharedIdx] + sharedMemory2[sharedIdx];

        // Stop the timer and record elapsed time
        clock_t stop = clock();
        *elapsedTime = 1000.0f * (float)(stop - start) / CLOCKS_PER_SEC;
    }
}

int main() {
    // Timer measure time spent on a process
    Timer timer = createTimer();

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

    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;

    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(int));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(int));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
        blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
    }

    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Allocate memory for elapsed time on the GPU
    float* device_elapsedTime;
    cudaMalloc((void**)&device_elapsedTime, sizeof(float));

    // Measure execution time for matrixAdditionSharedMemory
    float elapsedTime;
    beginTimer(timer);
    matrixAdditionSharedMemory1D << <gridDim, blockDim >> > (device_M1, device_M2, device_M3, device_elapsedTime);
    cudaDeviceSynchronize();
    endTimer(timer, "matrixAdditionSharedMemory1D", device_elapsedTime, &elapsedTime);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Copy the elapsed time from device to host
    cudaMemcpy(&elapsedTime, device_elapsedTime, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Elapsed Time: %.6f ms\n", elapsedTime);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);
    cudaFree(device_elapsedTime);

    return 0;
}
