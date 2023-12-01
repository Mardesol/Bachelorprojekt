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



//Pivoting kernels
__global__ void PivotAndSwap(float* A, int* pivotIndices, int n, int i) {
    // Find the pivot: maximum element in the current column
    int maxIndex = i;
    float maxValue = abs(A[i * n + i]);

    for (int row = i + 1; row < n; ++row) {
        float value = abs(A[row * n + i]);
        if (value > maxValue) {
            maxIndex = row;
            maxValue = value;
        }
    }

    pivotIndices[i] = maxIndex;

    // Swap rows if necessary
    if (maxIndex != i) {
        for (int col = 0; col < n; ++col) {
            float temp = A[i * n + col];
            A[i * n + col] = A[maxIndex * n + col];
            A[maxIndex * n + col] = temp;
        }
    }
}

__global__ void FindPivot(float* A, int* pivotIndices, int n, int i) {
    int maxIndex = i;
    float maxValue = abs(A[i * n + i]);

    for (int row = i + 1; row < n; ++row) {
        float value = abs(A[row * n + i]);
        if (value > maxValue) {
            maxIndex = row;
            maxValue = value;
        }
    }
    pivotIndices[i] = maxIndex;
}

__global__ void SwapRows(float* A, int* pivotIndices, int n, int i) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int col = tid; col < n; col += blockDim.x * gridDim.x) {
        float temp = A[i * n + col];
        A[i * n + col] = A[pivotIndices[i] * n + col];
        A[pivotIndices[i] * n + col] = temp;
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

int* Parallel_Pivoted(float* A, int n, dim3 blockDim) {

    int* pivotIndices;
    cudaMalloc(&pivotIndices, n * sizeof(int));

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        // Find pivot and swap

        //FindPivot << <1,1>> > (A, pivotIndices, n, i);
        dim3 blockDimRow(blockDim.x, 1);
        dim3 gridDimRow((n + blockDim.x - 1) / blockDim.x, 1);
        if (i == 0 || i == n - 1) { printf("launching pivot with a blockdim of %d, %d, and a gridDim of %d, %d\n", blockDim.x, 1, (n + blockDim.x - 1) / blockDim.x, 1); }
        //SwapRows << <gridDimRow, blockDimRow >> > (A, pivotIndices, n, i);
        PivotAndSwap << <gridDimRow, blockDimRow >> > (A, pivotIndices, n, i);
        //PivotAndSwap << <1, 1>> > (A, pivotIndices, n, i);
        cudaDeviceSynchronize();

        //Dimensions of the submatrix below/to the right of element (i,i)
        int subMatrixDim = n - i - 1;

        // Calculates the L values for row j
        dim3 blockDimColumn(1, blockDim.y);
        dim3 gridDimColumn(1, (subMatrixDim + blockDim.x - 1) / blockDim.x);
        ComputeLowerColumn << <gridDimColumn, blockDimColumn >> > (A, n, i);
        // No difference in result by dropping these, but faster performance
        //cudaDeviceSynchronize();

        dim3 gridDimSubmatrix((subMatrixDim + blockDim.x - 1) / blockDim.x, (subMatrixDim + blockDim.y - 1) / blockDim.y);
        UpdateSubmatrix << <gridDimSubmatrix, blockDim >> > (A, n, i);
        //cudaDeviceSynchronize();

    }

    int* hostPivotIndices = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostPivotIndices, pivotIndices, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(pivotIndices);
    return hostPivotIndices;
}







//Shared Memory kernels and main function

__global__ void PivotAndSwapShared(float* A, int* pivotIndices, int n, int i) {
    __shared__ float sharedRowA[32];
    __shared__ float sharedRowB[32];

    int tid = threadIdx.x;
    int maxIndex = pivotIndices[i]; // Assuming pivotIndices is precalculated

    // Ensure the thread ID is within the bounds of the row
    if (tid < n) {
        // Load the i-th and maxIndex-th rows into shared memory
        sharedRowA[tid] = A[i * n + tid];
        sharedRowB[tid] = A[maxIndex * n + tid];
    }

    // Wait for all threads in the block to load their elements
    __syncthreads();

    // Swap the elements in shared memory
    if (tid < n && maxIndex != i) {
        A[i * n + tid] = sharedRowB[tid];
        A[maxIndex * n + tid] = sharedRowA[tid];
    }
}

__global__ void ComputeLowerColumnShared(float* A, int n, int i) {
    __shared__ float pivotElement;
    pivotElement = A[i * n + i];

    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;

    if (row < n) {
        A[row * n + i] /= pivotElement;
    }
}

__global__ void UpdateSubmatrixShared(float* A, int n, int i) {
    __shared__ float sharedRow[32];
    __shared__ float sharedCol[32];

    int row = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + i + 1;

    if (row < n && threadIdx.x == 0) {
        sharedCol[threadIdx.y] = A[row * n + i];
    }
    if (col < n && threadIdx.y == 0) {
        sharedRow[threadIdx.x] = A[i * n + col];
    }

    __syncthreads();

    if (row < n && col < n) {
        A[row * n + col] -= sharedRow[threadIdx.x] * sharedCol[threadIdx.y];
    }
}

int* SharedMemory_Pivoted(float* A, int n, dim3 blockDim) {
    int* pivotIndices;
    cudaMalloc(&pivotIndices, n * sizeof(int));

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        // Invoke PivotAndSwap kernel
        PivotAndSwap << <1, 1 >> > (A, pivotIndices, n, i);
        cudaDeviceSynchronize();

        //Dimensions of the submatrix below/to the right of element (i,i)
        int subMatrixDim = n - i - 1;

        // Calculates the L values for row j
        dim3 blockDimColumn(1, blockDim.y);
        dim3 gridDimColumn(1, (subMatrixDim + blockDim.x - 1) / blockDim.x);
        ComputeLowerColumnShared << <gridDimColumn, blockDimColumn >> > (A, n, i);
        // No difference in result by dropping these, but faster performance
        //cudaDeviceSynchronize();

        dim3 gridDimSubmatrix((subMatrixDim + blockDim.x - 1) / blockDim.x, (subMatrixDim + blockDim.y - 1) / blockDim.y);
        UpdateSubmatrixShared << <gridDimSubmatrix, blockDim >> > (A, n, i);
        //cudaDeviceSynchronize();

    }

    int* hostPivotIndices = (int*)malloc(n * sizeof(int));
    cudaMemcpy(hostPivotIndices, pivotIndices, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(pivotIndices);
    return hostPivotIndices;
}

