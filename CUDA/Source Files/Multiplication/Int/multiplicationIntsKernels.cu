#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Matrix\matrixInts.cu"
#include "..\..\Timer\timer.cu"

__global__ void Sequential(int* M1, int* M2, int* M3) {
	for (int i = 0; i < M1Rows; i++) {
		for (int j = 0; j < M2Cols; j++) {
			int sum = 0;
			for (int k = 0; k < M1Cols; k++) {
				sum += M1[i * M1Cols + k] * M2[k * M2Cols + j];
			}
			M3[i * M2Cols + j] = sum;
		}
	}
}

__global__ void Parallel(int* M1, int* M2, int* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M1Rows && col < M2Cols) {
		int sum = 0;

		for (int i = 0; i < M1Cols; i++) {
			sum += M1[row * M1Cols + i] * M2[i * M2Cols + col];
		}
		M3[row * M2Cols + col] = sum;
	}
}

//__global__ void SharedMemoryAndTiling(int* M1, int* M2, int* M3) {
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Allocate shared memory
//	__shared__ int sharedMemory1[256];
//	__shared__ int sharedMemory2[256];
//
//	int sharedIndex1 = threadIdx.y * blockDim.x + threadIdx.x;
//	int sharedIndex2 = threadIdx.x * blockDim.y + threadIdx.y;
//
//	int sum = 0;
//
//	// Read into shared memory in a coalescing manner
//	for (int i = 0; i < M1Cols; i += blockDim.x) {
//		sharedMemory1[sharedIndex1] = (row < M1Rows && i + threadIdx.x < M1Cols) ? M1[row * M1Cols + i + threadIdx.x] : 0;
//		sharedMemory2[sharedIndex2] = (i + threadIdx.y < M1Cols && col < M2Cols) ? M2[(i + threadIdx.y) * M2Cols + col] : 0;
//
//		__syncthreads();
//
//		// Perform the multiplication
//		for (int j = 0; j < blockDim.x; j++) {	
//			sum += sharedMemory1[threadIdx.y * blockDim.x + j] * sharedMemory2[j * blockDim.y + threadIdx.x];
//		}
//
//		__syncthreads();
//	}
//
//	if (row < M1Rows && col < M2Cols) {
//		M3[row * M2Cols + col] = sum;
//	}
//}

__global__ void SharedMemoryAndTiling(int* M1, int* M2, int* M3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedMemory1[256];
    __shared__ int sharedMemory2[256];

    int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

    int sum = 0;

    for (int i = 0; i < M1Cols; i += blockDim.x) {
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
        for (int j = 0; j < numIterations; j++) {
            sum += sharedMemory1[threadIdx.y * blockDim.x + j] * sharedMemory2[j * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M1Rows && col < M2Cols) {
        M3[row * M2Cols + col] = sum;
    }
}

