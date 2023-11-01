#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

#include "..\Timer\timer.cu"
#include "..\Matrix\matrix.cu"

__global__ void Sequential(float* A, int n) {
    // Loop over each row
    for (int i = 0; i < n; i++) {
        
        // Compute U elements (upper triangular part)
        for (int j = i; j < n; j++) {
            
            float sum = A[i * n + j];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
            
            float sum = A[j * n + i];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            // Divide by the diagonal element
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}

__global__ void Sequential_Partial_Pivoting(float* A, int n) {
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

        __syncthreads();

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

// __global__ void LUD_Block(float* A, int n) {
    
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x; 
//     int ty = threadIdx.y;
//     int bdx = blockDim.x;
//     int bdy = blockDim.y;
//     const int blockSize = 4;

//     __shared__ float tile[blockSize][blockSize];

//     // Load tile into shared memory
// 	int row = by * bdy + ty;
// 	int col = bx * bdx + tx;

//     if(row < n && col < n) {
//         tile[ty][tx] = A[row * n + col];
//     }
//     __syncthreads();

//     // Perform LUD on the tile
//     for (int k = 0; k < blockSize; k++) {
//         if (tx == k && ty > k) {
//             tile[ty][tx] /= tile[k][k];
//         }
//         __syncthreads();
//         if (ty > k) {
//             for (int j = k+1; j < blockSize; j++) {
//                 if (tx == j) {
//                     tile[ty][tx] -= tile[ty][k] * tile[k][j];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     // Write tile back to global memory
//     if(row < n && col < n) {
//         A[row * n + col] = tile[ty][tx];
//     }
// }

__global__ void LUD_Block(float* A, int n) {
    const int blockSize = 16; // Assuming block size is 16x16
    __shared__ float tile[blockSize][blockSize];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockSize + ty;
    int col = bx * blockSize + tx;

    // Load tile into shared memory
    if(row < n && col < n) {
        tile[ty][tx] = A[row * n + col];
    }
    __syncthreads();

    for (int k = 0; k < blockSize; k++) {
        // Diagonal elements
        if (tx == k && ty == k) {
            for (int j = k + 1; j < blockSize; j++) {
                tile[k][j] /= tile[k][k];
            }
        }
        __syncthreads();

        // Off-diagonal elements
        if (tx > k && ty > k) {
            tile[ty][tx] -= tile[ty][k] * tile[k][tx];
        }
        __syncthreads();
    }

    // Write tile back to global memory
    if(row < n && col < n) {
        A[row * n + col] = tile[ty][tx];
    }
}

__global__ void LUD_Block_Similar(float* A, int n) {
    const int blockSize = 16; // Assuming block size is 16x16
    __shared__ float tile[blockSize][blockSize];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockSize + ty;
    int col = bx * blockSize + tx;

    // Load tile into shared memory
    if(row < n && col < n) {
        tile[ty][tx] = A[row * n + col];
    }
    __syncthreads();

    for (int i = 0; i < blockSize; i++) {
        float sum = 0;

        // Upper triangular matrix
        if (ty == i && tx >= i) {
            for (int j = 0; j < i; j++) {
                sum += tile[i][j] * tile[j][tx];
            }
            tile[i][tx] -= sum;
        }

        __syncthreads();

        // Lower triangular matrix
        if (ty > i && tx == i) {
            sum = 0;
            for (int j = 0; j < i; j++) {
                sum += tile[ty][j] * tile[j][i];
            }
            tile[ty][i] = (tile[ty][i] - sum) / tile[i][i];
        }

        __syncthreads();
    }

    // Write tile back to global memory
    if(row < n && col < n) {
        A[row * n + col] = tile[ty][tx];
    }
}
