#include "additionKernels.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Function to measure kernel execution time
float measureKernelExecutionTime(
    void (*kernel)(float *, float *, float *, int, int),
    float *M1, float *M2, float *M3, int MRows, int MCols,
    dim3 gridDim, dim3 blockDim)
{
    Timer timer = createTimer();
    beginTimer(timer);

    cudaDeviceSynchronize();
    kernel<<<gridDim, blockDim>>>(M1, M2, M3, MRows, MCols);
    cudaDeviceSynchronize();

    return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
    float *executionTimes,
    void (*kernel)(float *, float *, float *, int, int),
    float *M1, float *M2, float *M3, int MRows, int MCols,
    dim3 gridDim, dim3 blockDim)
{
    for (int i = 0; i < 100; i++)
    {
        // Measure execution time for the kernel
        float time = measureKernelExecutionTime(kernel, M1, M2, M3, MRows, MCols, gridDim, blockDim);
        executionTimes[i] = time;
    }
}

int main(int argc, char* argv[])
{
    int MRows = atoi(argv[1]);
    int MCols = atoi(argv[2]);
    size_t memorySize = MRows * MCols * sizeof(float);

    // Timer measure time spent on a process
    Timer timer = createTimer();

    beginTimer(timer);
    Matrix M1, M2, M3;
    float *device_M1, *device_M2, *device_M3;
    initializeMatricesAndMemory(M1, M2, M3, MRows, MCols, MRows, MCols, MRows, MCols);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize, memorySize, memorySize);
    copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize, memorySize);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (MRows <= 16 && MCols <= 16)
    {
        blockDim = dim3(MCols, MRows); // Use matrix size for smaller matrices
    }

    dim3 gridDim((MCols + blockDim.x - 1) / blockDim.x, (MRows + blockDim.y - 1) / blockDim.y);

    // Create an array to store execution times for each kernel
    float executionTimes[3][100]; // 3 kernels, 100 executions each

    // Measure and record execution times for all kernels
    measureExecutionTimes(executionTimes[0], Sequential,    device_M1, device_M2, device_M3, MRows, MCols, 1, 1);
    measureExecutionTimes(executionTimes[1], Parallel,      device_M1, device_M2, device_M3, MRows, MCols, gridDim, blockDim);
    measureExecutionTimes(executionTimes[2], SharedMemory,  device_M1, device_M2, device_M3, MRows, MCols, gridDim, blockDim);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, memorySize, cudaMemcpyDeviceToHost);

    // Open a new file to write the result into
    char fileName[100];                                                                     // Max length of filename
    sprintf(fileName, "Test/Floats_Execution_Times_Matrix_Size_%dx%d.csv", MRows, MCols);   // Customize filename to reflect size of result matrix
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return 1;
    }

    // Write execution times to the output file in separate columns
    fprintf(outputFile, "Sequential,Parallel,SharedMemory\n");
    for (int i = 0; i < 100; i++)
    {
        fprintf(outputFile, "%f,%f,%f\n",
                executionTimes[0][i],
                executionTimes[1][i],
                executionTimes[2][i]);
    }

    // Close the output file
    fclose(outputFile);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}