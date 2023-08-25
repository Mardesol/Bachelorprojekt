#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel to add two matrices in parallel
__global__ void matrixAddition(int *M1, int *M2, int *M3, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int index = row * N + col;
        M3[index] = M1[index] + M2[index];
    }
}

int main() {
    int N = 1000;                // Length of rows and cols
    char file[] = "1000.txt";    // Name of file
    
    // Open first matrix file
    FILE *file1 = fopen(file, "r");
    if (file1 == NULL) {
        perror("Unable to open file");
        return 1;
    }
    
    // Open second matrix file
    FILE *file2 = fopen(file, "r");
    if (file2 == NULL) {
        perror("Unable to open file");
        return 1;
    }

    // Allocate memory for matrices on the CPU
    int *host_M1 = (int *)malloc(N * N * sizeof(int));
    int *host_M2 = (int *)malloc(N * N * sizeof(int));
    int *host_M3 = (int *)malloc(N * N * sizeof(int));

    // Read data into host_M1 and host_M2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file1, "%d", &host_M1[i * N + j]);
            fscanf(file2, "%d", &host_M2[i * N + j]);
        }
    }

    // Close files
    fclose(file1);
    fclose(file2);

    // Allocate memory for matrices on the GPU
    int *device_M1, *device_M2, *device_M3;
    cudaMalloc((void **)&device_M1, N * N * sizeof(int));
    cudaMalloc((void **)&device_M2, N * N * sizeof(int));
    cudaMalloc((void **)&device_M3, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, host_M1, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, host_M2, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Start measuring time for matrix addition on the GPU
    clock_t begin = clock();

    // Launch the CUDA kernel to perform matrix addition
    matrixAddition<<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, N);

    // End measuring time for matrix addition on the GPU
    clock_t end = clock();
    double time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent on addition (GPU): %f seconds\n", time_spent2);

    // Copy the result matrix from device to host
    cudaMemcpy(host_M3, device_M3, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Open a new file to write the result into
    FILE *outputFile = fopen("result.txt", "w");
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
