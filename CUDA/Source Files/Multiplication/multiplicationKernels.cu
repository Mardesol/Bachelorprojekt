#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\Matrix\matrix.cu"
#include "..\Timer\timer.cu"

const int BLOCK_SIZE = 16;

__global__ void Sequential(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
	for (int i = 0; i < M1Rows; i++)
	{
		for (int j = 0; j < M2Cols; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < M1Cols; k++)
			{
				sum += M1[i * M1Cols + k] * M2[k * M2Cols + j];
			}
			M3[i * M2Cols + j] = sum;
		}
	}
}

__global__ void Parallel(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M1Rows && col < M2Cols)
	{
		float sum = 0.0f;

		for (int i = 0; i < M1Cols; i++)
		{
			sum += M1[row * M1Cols + i] * M2[i * M2Cols + col];
		}
		M3[row * M2Cols + col] = sum;
	}
}

__global__ void SharedMemoryAndTiling(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sharedMemory1[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float sharedMemory2[BLOCK_SIZE * BLOCK_SIZE];

	int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

	float sum = 0.0f;

	for (int i = 0; i < M1Cols; i += blockDim.x)
	{
		// Load M1 into shared memory
		if (row < M1Rows && (i + threadIdx.x) < M1Cols)
			sharedMemory1[sharedIndex] = M1[row * M1Cols + i + threadIdx.x];
		else
			sharedMemory1[sharedIndex] = 0;

		// Load M2 into shared memory
		if ((i + threadIdx.y) < M1Cols && col < M2Cols)
			sharedMemory2[sharedIndex] = M2[(i + threadIdx.y) * M2Cols + col];
		else
			sharedMemory2[sharedIndex] = 0;

		__syncthreads();

		// Tile multiplication
		int numIterations = (M1Cols - i > blockDim.x) ? blockDim.x : M1Cols - i;
		for (int j = 0; j < numIterations; j++)
		{
			sum += sharedMemory1[threadIdx.y * blockDim.x + j] * sharedMemory2[j * blockDim.x + threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M1Rows && col < M2Cols)
	{
		M3[row * M2Cols + col] = sum;
	}
}

__global__ void SharedMemoryAndTiling_32_32(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sharedMemory1[1024];
	__shared__ float sharedMemory2[1024];

	int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

	float sum = 0.0f;

	for (int i = 0; i < M1Cols; i += blockDim.x)
	{
		// Load M1 into shared memory
		if (row < M1Rows && (i + threadIdx.x) < M1Cols)
			sharedMemory1[sharedIndex] = M1[row * M1Cols + i + threadIdx.x];
		else
			sharedMemory1[sharedIndex] = 0;

		// Load M2 into shared memory
		if ((i + threadIdx.y) < M1Cols && col < M2Cols)
			sharedMemory2[sharedIndex] = M2[(i + threadIdx.y) * M2Cols + col];
		else
			sharedMemory2[sharedIndex] = 0;

		__syncthreads();

		// Tile multiplication
		int numIterations = (M1Cols - i > blockDim.x) ? blockDim.x : M1Cols - i;
		for (int j = 0; j < numIterations; j++)
		{
			sum += sharedMemory1[threadIdx.y * blockDim.x + j] * sharedMemory2[j * blockDim.x + threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M1Rows && col < M2Cols)
	{
		M3[row * M2Cols + col] = sum;
	}
}

__global__ void SharedMemory2DAndTiling(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedMemory1[16][16];  // Use 2D shared memory
    __shared__ float sharedMemory2[16][16];  // Use 2D shared memory

    float sum = 0.0f;

    for (int i = 0; i < M1Cols; i += blockDim.x)
    {
        // Load M1 into shared memory
        if (row < M1Rows && (i + threadIdx.x) < M1Cols)
            sharedMemory1[threadIdx.y][threadIdx.x] = M1[row * M1Cols + i + threadIdx.x];
        else
            sharedMemory1[threadIdx.y][threadIdx.x] = 0;

        // Load M2 into shared memory
        if ((i + threadIdx.y) < M1Cols && col < M2Cols)
            sharedMemory2[threadIdx.y][threadIdx.x] = M2[(i + threadIdx.y) * M2Cols + col];
        else
            sharedMemory2[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Tile multiplication
        int numIterations = (M1Cols - i > blockDim.x) ? blockDim.x : M1Cols - i;
        for (int j = 0; j < numIterations; j++)
        {
            sum += sharedMemory1[threadIdx.y][j] * sharedMemory2[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M1Rows && col < M2Cols)
    {
        M3[row * M2Cols + col] = sum;
    }
}

__global__ void SharedMemory2DAndTiling_32_32(float *M1, float *M2, float *M3, int M1Rows, int M1Cols, int M2Cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedMemory1[32][32];  // Use 2D shared memory
    __shared__ float sharedMemory2[32][32];  // Use 2D shared memory

    float sum = 0.0f;

    for (int i = 0; i < M1Cols; i += blockDim.x)
    {
        // Load M1 into shared memory
        if (row < M1Rows && (i + threadIdx.x) < M1Cols)
            sharedMemory1[threadIdx.y][threadIdx.x] = M1[row * M1Cols + i + threadIdx.x];
        else
            sharedMemory1[threadIdx.y][threadIdx.x] = 0;

        // Load M2 into shared memory
        if ((i + threadIdx.y) < M1Cols && col < M2Cols)
            sharedMemory2[threadIdx.y][threadIdx.x] = M2[(i + threadIdx.y) * M2Cols + col];
        else
            sharedMemory2[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Tile multiplication
        int numIterations = (M1Cols - i > blockDim.x) ? blockDim.x : M1Cols - i;
        for (int j = 0; j < numIterations; j++)
        {
            sum += sharedMemory1[threadIdx.y][j] * sharedMemory2[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M1Rows && col < M2Cols)
    {
        M3[row * M2Cols + col] = sum;
    }
}