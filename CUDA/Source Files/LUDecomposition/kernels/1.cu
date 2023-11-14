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

void LUD_Sequential(float **A, int n) {
    // Loop over each row
    for (int i = 0; i < n; i++) {

        // Compute U elements (upper triangular part)
        for (int j = i; j < n; j++) {

            float sum = A[i][j];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
        
            float sum = A[j][i];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            // Divide by the diagonal element
            A[j][i] = sum / A[i][i];
        }
    }
}

void LUD_Sequential(float **A, int n) {
    printf("before LUD \n");
    printf("n: %d \n", n);
    
    for (int i = 1; i <= n; i++) {
        printf("i: %d \n", i);
        
        for (int j = i; j <= n; j++) {
            printf("j1: %d \n", j);
            
            float sum = A[i][j];
            
            for (int k = 1; k < i; k++) {
                printf("k1: %d \n", k);
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        printf("after j loop \n");

        for (int j = i+1; j <= n; j++) {
            printf("j2: %d \n", j);
            float sum = A[j][i];
            
            for (int k = 1; k < i; k++) {
                printf("k2: %d \n", k);
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
    printf("after LUD \n");
}


void LUD_Sequential_Partial_Pivoting(float** A, int n) {
    for (int i = 1; i <= n; i++) {

        //Find pivot row                                    // Pivot row = row with the highest absolute value on the diagonal of the unworked rows
        int pivotRow = i;                                   //Set pivot row to current row
        float maxVal = fabs(A[i][i]);                       //Set max value to current rows diagonal
        for (int p = i + 1; p <= n; p++) {                  //Check if another row below has a higher absolut value on the diagonal
            if (fabs(A[p][i]) > maxVal) {
                maxVal = fabs(A[p][i]);                     //If yes, set that element to new max
                pivotRow = p;                               //And that row to the pivot row
            }
        }

        //Swap rows if needed
        if (pivotRow != i) {                                //Checks if current row is not the pivot row
            for (int j = 1; j <= n; j++) {                  //If not, swap the current row with the pivot row
                float temp = A[i][j];                       
                A[i][j] = A[pivotRow][j];
                A[pivotRow][j] = temp;
            }
        }

        //Perform LUD
        for (int j = i; j <= n; j++) {

            float sum = A[i][j];
            for (int k = 1; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        for (int j = i + 1; j <= n; j++) {

            float sum = A[j][i];
            for (int k = 1; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
}





//Copied from lud.cu
void LUD_cuSolver(float* device_A, int ADim, Timer timer) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int* d_pivot, * d_info;
    cudaMalloc((void**)&d_pivot, ADim * sizeof(int));
    cudaMalloc((void**)&d_info, sizeof(int));

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, ADim, ADim, device_A, ADim, &lwork);
    float* work;
    cudaMalloc((void**)&work, lwork * sizeof(float));

    beginTimer(timer);
    cusolverDnSgetrf(handle, ADim, ADim, device_A, ADim, work, NULL, d_info);
    cudaDeviceSynchronize();
    endTimer(timer, "cuSolver LUD (GPU)", printDebugMessages);

    // Clean up
    cudaFree(d_pivot);
    cudaFree(d_info);
    cudaFree(work);
    cusolverDnDestroy(handle);
}

void LUD_cuBLAS_Single(float* device_A, int ADim, Timer timer) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int* d_info;
    cudaMalloc((void**)&d_info, sizeof(int));

    // Wrap the pointer to device_A in an array for batched processing with batch size 1
    float* device_A_array[1] = { device_A };

    beginTimer(timer);
    cublasSgetrfBatched(handle, ADim, device_A_array, ADim, NULL, d_info, 1);
    cudaDeviceSynchronize();
    endTimer(timer, "cuBLAS LUD Single (GPU)", printDebugMessages);

    // Clean up
    cudaFree(d_info);
    cublasDestroy(handle);
}

