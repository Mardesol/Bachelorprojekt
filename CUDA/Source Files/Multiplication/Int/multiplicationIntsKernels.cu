#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Matrix\matrixInts.cu"
#include "..\..\Timer\timer.cu"

const int M1Rows = 100;
const int M1Cols = 100;
const int M2Rows = 100;
const int M2Cols = 100;
const int M3Rows = M1Rows;
const int M3Cols = M2Cols;

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

__global__ void SharedMemoryAndTiling(int* M1, int* M2, int* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ int sharedMemory1[32 * 32];
	__shared__ int sharedMemory2[32 * 32];

	int sharedIndex1 = threadIdx.y * blockDim.x + threadIdx.x;
	int sharedIndex2 = threadIdx.x * blockDim.y + threadIdx.y;

	int sum = 0;

	// Read into shared memory in a coalescing manner
	for (int i = 0; i < M1Cols; i += blockDim.x) {
		sharedMemory1[sharedIndex1] = (row < M1Rows && i + threadIdx.x < M1Cols) ? M1[row * M1Cols + i + threadIdx.x] : 0;
		sharedMemory2[sharedIndex2] = (i + threadIdx.y < M1Cols && col < M2Cols) ? M2[(i + threadIdx.y) * M2Cols + col] : 0;

		__syncthreads();

		// Perform the multiplication
		for (int j = 0; j < blockDim.x; j++) {	
			sum += sharedMemory1[threadIdx.y * blockDim.x + j] * sharedMemory2[j * blockDim.y + threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M1Rows && col < M2Cols) {
		M3[row * M2Cols + col] = sum;
	}
}