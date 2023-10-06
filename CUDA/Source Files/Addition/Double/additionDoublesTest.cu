#include "additionDoublesKernels.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Function to measure kernel execution time
float measureKernelExecutionTime(
    void (*kernel)(double*, double*, double*),
    double* M1, double* M2, double* M3,
    dim3 gridDim, dim3 blockDim
) {
    Timer timer = createTimer();
    beginTimer(timer);

    cudaDeviceSynchronize();
    kernel <<<gridDim, blockDim >>> (M1, M2, M3);
    cudaDeviceSynchronize();

    return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
    float* executionTimes,
    void (*kernel)(double*, double*, double*),
    double* M1, double* M2, double* M3,
    dim3 gridDim, dim3 blockDim
) {
    for (int i = 0; i < 100; i++) {
        // Measure execution time for the kernel
        float time = measureKernelExecutionTime(kernel, M1, M2, M3, gridDim, blockDim);
        executionTimes[i] = time;
    }
}

int main() {
    if (!additionCheck(M1Rows, M1Cols, M2Rows, M2Cols)) {
        perror("Matrices must have the same size");
        return 1;
    }
    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Create the matrix objects
    MatrixD M1 = createMatrixDoubles(M1Rows, M1Cols);
    MatrixD M2 = createMatrixDoubles(M2Rows, M2Cols);
    MatrixD M3 = createMatrixDoubles(M3Rows, M3Cols);

    // Populate the matrices
    populateWithRandomDoubles(M1);
    populateWithRandomDoubles(M2);

    // Stop the setup timer
    endTimer(timer, "setup", printDebugMessages);

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Allocate memory for matrices on the GPU
    double* device_M1, * device_M2, * device_M3;

    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(double));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(double));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(double));

    // Copy input matrices from host to device
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(double), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)", printDebugMessages);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
        blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
    }

    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Create an array to store execution times for each kernel
    float executionTimes[4][100]; // 4 kernels, 100 executions each

    // Measure and record execution times for all kernels
    measureExecutionTimes(executionTimes[0], Sequential,    device_M1, device_M2, device_M3, gridDim, blockDim);
    measureExecutionTimes(executionTimes[1], Parallel,      device_M1, device_M2, device_M3, gridDim, blockDim);
    measureExecutionTimes(executionTimes[2], SharedMemory,  device_M1, device_M2, device_M3, gridDim, blockDim);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // Open a new file to write the result into
    char fileName[100];                                                                                             // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Double_Execution_Times_Matrix_Size_%dx%d.csv", M3Rows, M3Cols);                            // Customize filename to reflect size of result matrix
    FILE* outputFile = fopen(fileName, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write execution times to the output file in separate columns
    fprintf(outputFile, "Sequential,Parallel,SharedMemory\n");
    for (int i = 0; i < 100; i++) {
        fprintf(outputFile, "%lf,%lf,%lf\n",
            executionTimes[0][i],
            executionTimes[1][i],
            executionTimes[2][i]);
    }

    // Close the output file
    fclose(outputFile);

    // Exit program
    return 0;
}