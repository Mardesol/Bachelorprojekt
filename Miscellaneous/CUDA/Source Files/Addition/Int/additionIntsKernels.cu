#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Timer\timer.cu"
#include "..\..\Matrix\matrixInts.cu"

// CUDA kernel to add two matrices sequentially
__global__ void Sequential(int *M1, int *M2, int *M3, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            M3[rows * i + j] = M1[rows * i + j] + M2[rows * i + j];
        }
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism
__global__ void Parallel(int *M1, int *M2, int *M3, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * rows + col;
        M3[index] = M1[index] + M2[index];
    }
}

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism, as well as shared memory
__global__ void SharedMemory(int *M1, int *M2, int *M3, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedMemory1[32 * 32];
    __shared__ int sharedMemory2[32 * 32];

    int index = row * rows + col;
    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

    // Ensure the index is within the matrix dimensions before loading into shared memory
    if (index < rows * cols)
    {
        sharedMemory1[sharedIndex] = M1[index];
        sharedMemory2[sharedIndex] = M2[index];
    }

    __syncthreads(); // Ensure all threads have loaded data

    if (index < rows * cols)
    {
        M3[index] = sharedMemory1[sharedIndex] + sharedMemory2[sharedIndex];
    }
}