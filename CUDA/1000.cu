#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// CUDA kernel to add two matrices
__global__ void matrixAdditionSimple(int* M1, int* M2, int* M3, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N * N) {
        M3[index] = M1[index] + M2[index];
    }
}

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
    // Variables to measure time spent on a process
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float setupTime;
    float hostToDeviceTime;
    float deviceToHostTime;
    float calculationTime;
    //float shutdownTime;

    // Start the setup timer
    cudaEventRecord(start, 0);

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

    // Stop the setup timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setupTime, start, stop);
    printf("Time spent on setup:                      %f seconds\n", setupTime);


    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    cudaEventRecord(start, 0);

    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;
    cudaMalloc((void**)&device_M1, N * N * sizeof(int));
    cudaMalloc((void**)&device_M2, N * N * sizeof(int));
    cudaMalloc((void**)&device_M3, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, host_M1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, host_M2, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&hostToDeviceTime, start, stop);
    printf("Time spent on data transfer (CPU -> GPU): %f seconds\n", hostToDeviceTime);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Start the matrix addition timer
    cudaEventRecord(start, 0);

    // Launch the CUDA kernel to perform matrix addition
    matrixAdditionSimple <<<gridDim, blockDim>>> (device_M1, device_M2, device_M3, N);

    // Stop the matrix addition timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&calculationTime, start, stop);
    printf("Time spent on matrix addition (GPU):      %f seconds\n", calculationTime);

    // Start the data transfer timer (GPU -> CPU / Device -> Host)
    cudaEventRecord(start, 0);

    // Copy the result matrix from device to host
    cudaMemcpy(host_M3, device_M3, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop the  data transfer timer (GPU -> CPU / Device -> Host)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&deviceToHostTime, start, stop);
    printf("Time spent on data transfer (GPU -> CPU): %f seconds\n", deviceToHostTime);

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
