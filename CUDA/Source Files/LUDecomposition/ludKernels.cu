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