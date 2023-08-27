#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel to add two matrices in parallel
__global__ void matrixAddition(int* M1, int* M2, int* M3, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        M3[index] = M1[index] + M2[index];
    }
}

int main() {
    // Start measuring time OS spends on process
    clock_t setupBegin = clock();

    int N = 1000;                // Length of rows and cols
    int pattern[] = { 1, 2, 3 };

    // Allocate memory for matrices on the CPU
    int* host_M1 = (int*)malloc(N * N * sizeof(int));
    int* host_M2 = (int*)malloc(N * N * sizeof(int));
    int* host_M3 = (int*)malloc(N * N * sizeof(int));

    // Initialize matrices M1 and M2 with the pattern
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_M1[i * N + j] = pattern[j % 3];
            host_M2[i * N + j] = pattern[j % 3];
        }
    }

    // End measuring time OS spends on process
    clock_t setupEnd = clock();
    double time_spent1 = (double)(setupEnd - setupBegin) / CLOCKS_PER_SEC;
    printf("Time spent on setup: %f seconds\n", time_spent1);

    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;
    cudaMalloc((void**)&device_M1, N * N * sizeof(int));
    cudaMalloc((void**)&device_M2, N * N * sizeof(int));
    cudaMalloc((void**)&device_M3, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, host_M1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, host_M2, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Start measuring time for matrix addition on the GPU
    clock_t begin = clock();

    // Launch the CUDA kernel to perform matrix addition
    matrixAddition <<<gridDim, blockDim>>> (device_M1, device_M2, device_M3, N);

    // End measuring time for matrix addition on the GPU
    clock_t end = clock();
    double time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent on addition (GPU): %f seconds\n", time_spent2);

    // Copy the result matrix from device to host
    cudaMemcpy(host_M3, device_M3, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Open a new file to write the result into
    FILE* outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(outputFile, "%d ", host_M3[i * N + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the result file
    fclose(outputFile);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);
    free(host_M1);
    free(host_M2);
    free(host_M3);

    // End program
    return 0;
}
