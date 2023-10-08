#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Matrix\matrixFloats.cu"
#include "..\..\Timer\timer.cu"

__global__ void Sequential(float* M1, float* M2, float* M3) {
	for (int i = 0; i < M1Rows; i++) {
		for (int j = 0; j < M2Cols; j++) {
			float sum = 0.0f;
			for (int k = 0; k < M1Cols; k++) {
				sum += M1[i * M1Cols + k] * M2[k * M2Cols + j];
			}
			M3[i * M2Cols + j] = sum;
		}
	}
}

__global__ void Parallel(float* M1, float* M2, float* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M1Rows && col < M2Cols) {
		float sum = 0.0f;

		for (int i = 0; i < M1Cols; i++) {
			sum += M1[row * M1Cols + i] * M2[i * M2Cols + col];
		}
		M3[row * M2Cols + col] = sum;
	}
}

//__global__ void SharedMemoryAndTiling(float* M1, float* M2, float* M3) {
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//	// Allocate shared memory
//	__shared__ float sharedMemory1[32 * 32];
//	__shared__ float sharedMemory2[32 * 32];
//
//	int sharedIndex1 = threadIdx.y * blockDim.x + threadIdx.x;
//	int sharedIndex2 = threadIdx.x * blockDim.y + threadIdx.y;
//
//	float sum = 0.0f;
//
//	// Read into shared memory in a coalescing manner
//	for (int i = 0; i < M1Cols; i += blockDim.x) {
//		sharedMemory1[sharedIndex1] = (row < M1Rows && i + threadIdx.x < M1Cols) ? M1[row * M1Cols + i + threadIdx.x] : 0.0f;
//		sharedMemory2[sharedIndex2] = (i + threadIdx.y < M1Cols && col < M2Cols) ? M2[(i + threadIdx.y) * M2Cols + col] : 0.0f;
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

__global__ void SharedMemoryAndTiling(float* M1, float* M2, float* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sharedMemory1[256];
	__shared__ float sharedMemory2[256];

	int sharedIndex = threadIdx.y * blockDim.x + threadIdx.x;

	float sum = 0.0f;

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