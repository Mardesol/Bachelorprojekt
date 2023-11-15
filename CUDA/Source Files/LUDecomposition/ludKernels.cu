#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>

#include <stdio.h>
#include <stdlib.h>

#include "..\Timer\timer.cu"
#include "..\Matrix\matrix.cu"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cusolverDn.h>
#include <cuda_runtime_api.h>

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

__global__ void New_Sequential(float* A, int n) {
    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
        }
        // Compute U elements (upper triangular part)
        for (int j = i + 1; j < n; j++) {
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] = A[j * n + k] - A[i * n + k] * A[j * n + i];
            }
        }
    }
}

__global__ void New_Sequential_With_Partial_Pivoting(float* A, int n) {
    // Loop over each row - Must be done 1 at the time
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

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
        }
        // Compute U elements (upper triangular part)
        for (int j = i + 1; j < n; j++) {
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] = A[j * n + k] - A[i * n + k] * A[j * n + i];
            }
        }
    }
}

//Parallel kernels and main function

__global__ void ComputeLowerColumn(float* A, int n, int i) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;

    if (row < n) {
        A[row * n + i] = A[row * n + i] / A[i * n + i];
    }
}


__global__ void UpdateSubmatrix(float* A, int n, int i) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + i + 1;

    if (row < n && col < n) {
        A[row * n + col] = A[row * n + col] - A[i * n + col] * A[row * n + i];
    }
}

void Parallel(float* A, int n, dim3 blockDim) {

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        //Dimensions of the submatrix below/to the right of element (i,i)
        int subMatrixDim = n - i - 1;

        // Calculates the L values for row j
        dim3 blockDimColumn(1, blockDim.y);
        dim3 gridDimColumn(1, (subMatrixDim + blockDim.x - 1) / blockDim.x);
        ComputeLowerColumn << <gridDimColumn, blockDimColumn >> > (A, n, i);
        cudaDeviceSynchronize();
        
        dim3 gridDimSubmatrix((subMatrixDim + blockDim.x - 1) / blockDim.x, (subMatrixDim + blockDim.y - 1) / blockDim.y);
        UpdateSubmatrix << <gridDimSubmatrix, blockDim >> > (A, n, i);
        cudaDeviceSynchronize();

    }
}

//Shared Memory kernels and main function

__global__ void ComputeLowerColumnShared(float* A, int n, int i) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    int col = i;

    if (row < n) {
        __shared__ float sharedA[16 * 16];
        sharedA[threadIdx.y] = A[row * n + col];
        __syncthreads();

        A[row * n + col] = sharedA[threadIdx.y] / A[i * n + i];
    }
}

__global__ void UpdateSubmatrixShared(float* A, int n, int i) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + i + 1;

    if (row < n && col < n) {
        __shared__ float sharedA_row[16 * 16];
        __shared__ float sharedA_col[16 * 16];

        sharedA_row[threadIdx.y] = A[row * n + i];
        sharedA_col[threadIdx.x] = A[i * n + col];

        __syncthreads();

        A[row * n + col] -= sharedA_row[threadIdx.y] * sharedA_col[threadIdx.x];
    }
}

void SharedMemory(float* A, int n, dim3 blockDim) {

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        //Dimensions of the submatrix below/to the right of element (i,i)
        int subMatrixDim = n - i - 1;

        // Calculates the L values for row j
        dim3 blockDimColumn(1, blockDim.y);
        dim3 gridDimColumn(1, (subMatrixDim + blockDim.x - 1) / blockDim.x);
        ComputeLowerColumnShared << <gridDimColumn, blockDimColumn >> > (A, n, i);
        cudaDeviceSynchronize();

        dim3 gridDimSubmatrix((subMatrixDim + blockDim.x - 1) / blockDim.x, (subMatrixDim + blockDim.y - 1) / blockDim.y);
        UpdateSubmatrixShared << <gridDimSubmatrix, blockDim >> > (A, n, i);
        cudaDeviceSynchronize();

    }
}
