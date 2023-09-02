#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main() {
    // Variables to track time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 
    float setupTime;

    // Start measuring time OS spends on process
    
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

    // End measuring time OS spends on process
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&setupTime, start, stop);

    printf("Time spent on setup: %f seconds\n", setupTime);





    // Allocate memory for matrices on the GPU
    int* device_M1, * device_M2, * device_M3;
    cudaMalloc((void**)&device_M1, N * N * sizeof(int));
    cudaMalloc((void**)&device_M2, N * N * sizeof(int));
    cudaMalloc((void**)&device_M3, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, host_M1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, host_M2, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(256);  // Use a single thread block
    dim3 gridDim((N * N + blockDim.x - 1) / blockDim.x);


    // Start measuring time for matrix addition on the GPU
    float elapsedTime2;






    cudaEventRecord(start, 0);

    // Launch the CUDA kernel to perform matrix addition sequentially on GPU
    matrixAdditionSimple <<<gridDim, blockDim>>> (device_M1, device_M2, device_M3, N);

    // End measuring time for matrix addition on the GPU
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime2, start, stop);

    printf("Time spent on addition (GPU): %f seconds\n", elapsedTime2);

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
