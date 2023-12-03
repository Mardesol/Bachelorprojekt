#include "additionKernels.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Function to measure kernel execution time
float measureKernelExecutionTime(
    void (*kernel)(float *, float *, float *, int, int),
    Matrix CPU_M1, Matrix CPU_M2, Matrix CPU_M3, float *M1, float *M2, float *M3, int MRows, int MCols, size_t memorySize)
{
    Timer timer = createTimer();
    beginTimer(timer);

        allocateMemoryOnGPU(M1, M2, M3, memorySize, memorySize, memorySize);
        copyMatricesToGPU(CPU_M1, CPU_M2, M1, M2, memorySize, memorySize);

        // Define block and grid dimensions for CUDA kernel
        dim3 blockDim(16, 16);

        if (MRows <= 16 && MCols <= 16)
        {
            blockDim = dim3(MCols, MRows); // Use matrix size for smaller matrices
        }

        dim3 gridDim((MCols + blockDim.x - 1) / blockDim.x, (MRows + blockDim.y - 1) / blockDim.y);

        cudaDeviceSynchronize();
        kernel<<<gridDim, blockDim>>>(M1, M2, M3, MRows, MCols);
        cudaDeviceSynchronize();

        cudaMemcpy(CPU_M3.data, M3, memorySize, cudaMemcpyDeviceToHost);

    return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
    float *executionTimes,
    void (*kernel)(float *, float *, float *, int, int),
    Matrix CPU_M1, Matrix CPU_M2, Matrix CPU_M3, float *M1, float *M2, float *M3, int MRows, int MCols, size_t memorySize)
{
    for (int i = 0; i < 1; i++)
    {
        // Measure execution time for the kernel
        float time = measureKernelExecutionTime(kernel, CPU_M1, CPU_M2, CPU_M3, M1, M2, M3, MRows, MCols, memorySize);
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

    // Create an array to store execution times for each kernel
    float executionTimes[3][100]; // 3 kernels, 100 executions each

    // Measure and record execution times for all kernels
    //measureExecutionTimes(executionTimes[0], Sequential,    device_M1, device_M2, device_M3, MRows, MCols, gridDim, blockDim);
    measureExecutionTimes(executionTimes[1], Parallel,      M1, M2, M3, device_M1, device_M2, device_M3, MRows, MCols, memorySize);
    measureExecutionTimes(executionTimes[2], SharedMemory,  M1, M2, M3, device_M1, device_M2, device_M3, MRows, MCols, memorySize);

    // Open a new file to write the result into
    char fileName[100];                                                                     // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Floats_Execution_Times_Matrix_Size_%dx%d.csv", MRows, MCols);   // Customize filename to reflect size of result matrix
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return 1;
    }

    // Write execution times to the output file in separate columns
    fprintf(outputFile, "Sequential,Parallel,SharedMemory\n");
    for (int i = 0; i < 1; i++)
    {
        fprintf(outputFile, "%f, %f \n", //,%f,%f
                //executionTimes[0][i]);
                executionTimes[1][i],
                executionTimes[2][i]);
    }

    // Close the output file
    fclose(outputFile);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}