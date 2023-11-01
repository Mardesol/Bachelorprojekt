#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

#include "..\Timer\timer.cu"
#include "..\Matrix\matrixFloats.cu"

__global__ void LUD_Sequential(float* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        for (int j = i + 1; j < n; j++) {
            float sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}


__global__ void LUD_Sequential_Partial_Pivoting(float* A, int n) {
    for (int i = 0; i < n; i++) {

        // Find pivot row
        int pivotRow = i;
        float maxVal = fabsf(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabsf(A[p * n + i]) > maxVal) {
                maxVal = fabsf(A[p * n + i]);
                pivotRow = p;
            }
        }

        // Swap rows if needed
        if (pivotRow != i) {
            for (int j = 0; j < n; j++) {
                float temp = A[i * n + j];
                A[i * n + j] = A[pivotRow * n + j];
                A[pivotRow * n + j] = temp;
            }
        }

        // Perform LUD
        for (int j = i; j < n; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        for (int j = i + 1; j < n; j++) {
            float sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}


// __global__ void LUD_Parallel(float* A, int n) {

// }

__global__ void LUD_Block(float* A, int n) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    const int blockSize = 16;

    __shared__ float tile[blockSize][blockSize];

    // Load tile into shared memory
	int row = by * bdy + ty;
	int col = bx * bdx + tx;

    if(row < n && col < n) {
        tile[ty][tx] = A[row * n + col];
    }
    __syncthreads();

    // Perform LUD on the tile
    for (int k = 0; k < blockSize; k++) {
        if (tx == k && ty > k) {
            tile[ty][tx] /= tile[k][k];
        }
        __syncthreads();
        if (ty > k) {
            for (int j = k+1; j < blockSize; j++) {
                if (tx == j) {
                    tile[ty][tx] -= tile[ty][k] * tile[k][j];
                }
            }
        }
        __syncthreads();
    }

    // Write tile back to global memory
    if(row < n && col < n) {
        A[row * n + col] = tile[ty][tx];
    }
}
