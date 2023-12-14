#include "kernel.cu"
#include "..\Matrix\matrixCompatability.cu"

int main(int argc, char* argv[])
{
    int MRows = atoi(argv[1]);
    int MCols = atoi(argv[2]);
    size_t memorySize = MRows * MCols * sizeof(float);

    Timer timer = createTimer();

    Matrix M1, M2, M3;
    float *device_M1, *device_M2, *device_M3;
    float before, after;
    initializeMatricesAndMemory(M1, M2, M3, MRows, MCols, MRows, MCols, MRows, MCols);
   
    // Create an array to store execution times for each kernel
    float transferTimes[100]; // 100 executions

    for (int i = 0; i < 100; i++) {
        // Measure (host -> device) transfer time including necesarry setup
        beginTimer(timer);

        allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize, memorySize, memorySize);
        copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize, memorySize);
        
        // Define block and grid dimensions for CUDA kernel
        dim3 blockDim(16, 16);
        if (MRows <= 16 && MCols <= 16) {blockDim = dim3(MCols, MRows);}
        dim3 gridDim((MCols + blockDim.x - 1) / blockDim.x, (MRows + blockDim.y - 1) / blockDim.y);
        
        before = endTimerReturnTime(timer); 

        cudaDeviceSynchronize();
        Parallel<<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, MRows, MCols);
        cudaDeviceSynchronize();

        beginTimer(timer);

        cudaMemcpy(M3.data, device_M3, memorySize, cudaMemcpyDeviceToHost); 
        freeDeviceMemory(device_M1, device_M2, device_M3);
      
        after = endTimerReturnTime(timer); 

        transferTimes[i] = before + after;
    }  

    // Open a new file to write the result into
    char fileName[100];                                                             // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Transfer_Times_Matrix_Size_%dx%d.csv", MRows, MCols);   // Customize filename to reflect size of result matrix
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return 1;
    }

    // Write execution times to the output file in separate columns
    fprintf(outputFile, "Transfer time\n");
    for (int i = 0; i < 100; i++)
    {
        fprintf(outputFile, "%f \n",
            transferTimes[i]);
    }

    // Close the output file
    fclose(outputFile);

    // Exit program
    return 0;
}