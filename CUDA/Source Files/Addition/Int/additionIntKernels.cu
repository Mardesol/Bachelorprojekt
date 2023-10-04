#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu" 
#include "..\..\Matrix\matrixInts.cu"

const int rows = 200;
const int cols = 200;

const int M1Rows = rows;
const int M2Rows = rows;
const int M3Rows = rows;

const int M3Cols = cols;
const int M1Cols = cols;
const int M2Cols = cols;

// CUDA kernel to add two matrices sequentially
__global__ void Sequential(int* M1, int* M2, int* M3) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M1Cols; j++) {
            M3[M1Rows * i + j] = M1[M1Rows * i + j] + M2[M1Rows * i + j];
        }
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism
__global__ void Parallel(int* M1, int* M2, int* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M3Rows && col < M3Cols) {
        int index = row * M3Rows + col;
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism, as well as shared memory
__global__ void SharedMemory(int* M1, int* M2, int* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedMemory1[32 * 32];
    __shared__ int sharedMemory2[32 * 32];

    int index = row * M3Rows + col;
    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

    // Ensure the index is within the matrix dimensions before loading into shared memory
    if (index < M3Rows * M3Cols) {
        sharedMemory1[sharedIndex] = M1[index];
        sharedMemory2[sharedIndex] = M2[index];
    }

    __syncthreads();  // Ensure all threads have loaded data

    if (index < M3Rows * M3Cols) {
        M3[index] = sharedMemory1[sharedIndex] + sharedMemory2[sharedIndex];
    }
}