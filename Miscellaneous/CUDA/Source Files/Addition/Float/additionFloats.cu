#include "additionFloatsKernels.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const size_t FILENAME_MAX_LENGTH = 256;

const char* executeChosenKernel(int KernelNumToPerform, float *device_M1, float *device_M2, float *device_M3, int MRows, int MCols, Timer timer)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((MCols + blockDim.x - 1) / blockDim.x, (MRows + blockDim.y - 1) / blockDim.y);
    const char *kernelName;

    switch (KernelNumToPerform)
    {
    case 1:
        kernelName = "Sequential";
        beginTimer(timer);
        Sequential<<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, MRows, MCols);
        endTimer(timer, "Sequential matrix addition (GPU)", printDebugMessages);
        break;
    case 2:
        kernelName = "Parallel";
        beginTimer(timer);
        Parallel<<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, MRows, MCols);
        endTimer(timer, "Parallel matrix addition (GPU)", printDebugMessages);
        break;
    case 3:
        kernelName = "SharedMemory";
        beginTimer(timer);
        SharedMemory<<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, MRows, MCols);
        endTimer(timer, "Shared memory matrix addition (GPU)", printDebugMessages);
        break;
    default:
        kernelName = "Unknown";
        break;
    }
    return kernelName;
}

int main(int argc, char *argv[])
{
    int KernelNumToPerform = atoi(argv[1]);
    int MRows = atoi(argv[2]);
    int MCols = atoi(argv[3]);
    size_t memorySize = MRows * MCols * sizeof(float);

    Timer timer = createTimer();

    beginTimer(timer);
    MatrixF M1, M2, M3;
    float *device_M1, *device_M2, *device_M3;
    initializeMatricesAndMemory(M1, M2, M3, MRows, MCols, MRows, MCols, MRows, MCols);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize, memorySize, memorySize);
    copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize, memorySize);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    const char* kernelName = executeChosenKernel(KernelNumToPerform, device_M1, device_M2, device_M3, MRows, MCols, timer);

    cudaMemcpy(M3.data, device_M3, memorySize, cudaMemcpyDeviceToHost);

    // Setup a CPU comparison matrix
    MatrixF MCPU = createMatrixFloats(MRows, MCols);
    additionFloats(M1, M2, MCPU);

    // Validate result by comparing to CPU calculations
    bool valid = compareMatricesFloats(MCPU, M3);
    if (valid)
    {
        printf("Matrix addition results match!\n");
    }
    else
    {
        printf("Matrix addition results do not match.\n");
        // Write the CPU Matrix to text file for analysis
        char fileNameCPU[100];
        sprintf(fileNameCPU, "resultsFloatsCPU.txt");

        printMatrixToFileFloats(fileNameCPU, MCPU);
    }

    char fileName[FILENAME_MAX_LENGTH];
    sprintf(fileName, "Test/Addition_%s_Floats_Runtime_Matrix_Size_%dx%d.csv", kernelName, MRows, MCols);
    printMatrixToFileFloats(fileName, M3);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}