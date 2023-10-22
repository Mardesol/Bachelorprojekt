#include "additionDoublesKernels.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const size_t FILENAME_MAX_LENGTH = 256;

// Execute the chosen kernel
const char* executeChosenKernel(int KernelNumToPerform, double *device_M1, double *device_M2, double *device_M3, int MRows, int MCols, Timer timer)
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
    size_t memorySize = MRows * MCols * sizeof(double);

    Timer timer = createTimer();

    beginTimer(timer);
    MatrixD M1, M2, M3;
    double *device_M1, *device_M2, *device_M3;
    initializeMatricesAndMemory(M1, M2, M3, MRows, MCols, MRows, MCols, MRows, MCols);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize, memorySize, memorySize);
    copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize, memorySize);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);
                                                                         
    const char* kernelName = executeChosenKernel(KernelNumToPerform, device_M1, device_M2, device_M3, MRows, MCols, timer); // Execute the kernels chosen by input arguments

    cudaMemcpy(M3.data, device_M3, memorySize, cudaMemcpyDeviceToHost); // Copy the result matrix from device to host

    // Setup a CPU comparison matrix
    MatrixD MCPU = createMatrixDoubles(MRows, MCols);
    additionDoubles(M1, M2, MCPU);

    // Validate result by comparing to CPU calculations
    bool valid = compareMatricesDoubles(MCPU, M3);
    if (valid)
    {
        printf("Matrix addition results match!\n");
    }
    else
    {
        printf("Matrix addition results do not match.\n");
        // Write the CPU Matrix to text file for analysis
        char fileNameCPU[100];
        sprintf(fileNameCPU, "resultsDoublesCPU.txt");

        printMatrixToFileDoubles(fileNameCPU, MCPU);
    }

    char fileName[FILENAME_MAX_LENGTH];
    sprintf(fileName, "Test/Addition_%s_Doubles_Runtime_Matrix_Size_%dx%d.csv", kernelName, MRows, MCols);
    printMatrixToFileDoubles(fileName, M3);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}