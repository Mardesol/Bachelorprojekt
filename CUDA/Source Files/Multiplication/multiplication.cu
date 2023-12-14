#include "multiplicationKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Execute the chosen kernel
const char* executeChosenKernel(int KernelNumToPerform, float *device_M1, float *device_M2, float *device_M3, int M1Rows, int M1Cols, int M2Cols, Timer timer)
{
    dim3 blockDim_16(16, 16);
    dim3 gridDim((M1Cols + blockDim_16.x - 1) / blockDim_16.x, (M1Rows + blockDim_16.y - 1) / blockDim_16.y);

    dim3 blockDim_32(32, 32);
    dim3 gridDim_32((M1Cols + blockDim_32.x - 1) / blockDim_32.x, (M1Rows + blockDim_32.y - 1) / blockDim_32.y);

    const char *kernelName;

    switch (KernelNumToPerform)
    {
    case 1:
        kernelName = "Sequential";
        beginTimer(timer);
        Sequential<<<gridDim, blockDim_16>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "Sequential matrix multiplication (GPU)", printDebugMessages);
        break;
    
    case 2:
        kernelName = "Parallel";
        beginTimer(timer);
        Parallel<<<gridDim, blockDim_16>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "Parallel matrix multiplication (GPU)", printDebugMessages);
        break;
    
    case 3:
        kernelName = "SharedMemoryAndTiling";
        beginTimer(timer);
        SharedMemoryAndTiling<<<gridDim, blockDim_16>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "SharedMemoryAndTiling matrix multiplication (GPU)", printDebugMessages);
        break;
    
    case 4:
        kernelName = "SharedMemoryAndTiling_32_32";
        beginTimer(timer);
        SharedMemoryAndTiling_32_32<<<gridDim_32, blockDim_32>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "SharedMemoryAndTiling matrix multiplication (GPU)", printDebugMessages);
        break;

    case 5:
        kernelName = "SharedMemory2DAndTiling";
        beginTimer(timer);
        SharedMemory2DAndTiling<<<gridDim, blockDim_16>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "SharedMemory2DAndTiling matrix multiplication (GPU)", printDebugMessages);
        break;
    
    case 6:
        kernelName = "SharedMemory2DAndTiling_32_32";
        beginTimer(timer);
        SharedMemory2DAndTiling_32_32<<<gridDim_32, blockDim_32>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols);
        endTimer(timer, "SharedMemory2DAndTiling_V2 matrix multiplication (GPU)", printDebugMessages);
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
    int M1Rows = atoi(argv[2]);
    int M1Cols = atoi(argv[3]);
	int M2Rows = atoi(argv[4]);
    int M2Cols = atoi(argv[5]);
	int M3Rows = M1Rows;
    int M3Cols = M2Cols;

    size_t memorySize1 = M1Rows * M1Cols * sizeof(int);
	size_t memorySize2 = M2Rows * M2Cols * sizeof(int);
	size_t memorySize3 = M3Rows * M3Cols * sizeof(int);

    if (!isCompatibleForMultiplication(M1Cols, M2Rows))
    {
        perror("Matrices must be compatible");
        return 1;
    }

    // Timer measure time spent on a process
    Timer timer = createTimer();

    beginTimer(timer);
    Matrix M1, M2, M3;
    float *device_M1, *device_M2, *device_M3;
    initializeMatricesAndMemory(M1, M2, M3, M1Rows, M1Cols, M2Rows, M2Cols, M3Rows, M3Cols);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize1, memorySize2, memorySize3);
    copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize1, memorySize2);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    const char* kernelName = executeChosenKernel(KernelNumToPerform, device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols, timer);

    // Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, memorySize3, cudaMemcpyDeviceToHost);

    // Setup a CPU comparison matrix
    Matrix MCPU = createMatrix(M3Rows, M3Cols);
    multiplication(M1, M2, MCPU);

    // Validate result by comparing to CPU calculations
    bool valid = compareMatrices(MCPU, M3);
    if (valid)
    {
        printf("Matrix multiplication results match!\n");
    }
    else
    {
        printf("Matrix multiplication results do not match.\n");
        // Write the CPU matrix to text file for analysis
        char fileNameCPU[100];
        sprintf(fileNameCPU, "results_CPU.txt");

        printMatrixToFile(fileNameCPU, MCPU);
    }

    // Open a new file to write the result into
    char fileName[100];                                                                                    // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Multiplication_%s_Runtime_Matrix_Size_%dx%d.csv", kernelName, M3Rows, M3Cols); // Customize filename to reflect size of result matrix
    printMatrixToFile(fileName, M3);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}