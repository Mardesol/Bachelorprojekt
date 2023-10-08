#include "additionIntsKernels.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const size_t FILENAME_MAX_LENGTH = 256;

// Execute the chosen kernel
const char* executeChosenKernel(int KernelNumToPerform, int* device_M1, int* device_M2, int* device_M3, Timer timer) {
    dim3 blockDim(32, 32);
    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);
    const char* kernelName;

    switch (KernelNumToPerform) {
    case 1:
        kernelName = "Sequential";
        beginTimer(timer);
        Sequential << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);
        endTimer(timer, "Sequential matrix addition (GPU)", printDebugMessages);
        break;
    case 2:
        kernelName = "Parallel";
        beginTimer(timer);
        Parallel << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);
        endTimer(timer, "Parallel matrix addition (GPU)", printDebugMessages);
        break;
    case 3:
        kernelName = "SharedMemory";
        beginTimer(timer);
        SharedMemory << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);
        endTimer(timer, "Shared memory matrix addition (GPU)", printDebugMessages);
        break;
    default:
        kernelName = "Unknown";
        break;
    }
    return kernelName;
}

int main(int argc, char* argv[]) {
    if (!isCompatibleForAddition(M1Rows, M1Cols, M2Rows, M2Cols)) {
        perror("Matrices must have the same size");
        return 1;
    }

    Timer timer = createTimer();

    beginTimer(timer);
    MatrixI M1, M2, M3;
    int* device_M1, * device_M2, * device_M3;
    initializeMatricesAndMemory(M1, M2, M3);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3);
    copyMatricesToGPU(M1, M2, device_M1, device_M2);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    int KernelNumToPerform = atoi(argv[1]);
    const char* kernelName = executeChosenKernel(KernelNumToPerform, device_M1, device_M2, device_M3, timer);

    cudaMemcpy(M3.data, device_M3, memorySize3, cudaMemcpyDeviceToHost);

    // Setup a CPU comparison matrix
    MatrixI MCPU = createMatrixInts(M3Rows, M3Cols);
    additionInts(M1, M2, MCPU);

    //Validate result by comparing to CPU calculations
    bool valid = compareMatricesInts(MCPU, M3);
    if (valid) {
        printf("Matrix addition results match!\n");
    }
    else {
        printf("Matrix addition results do not match.\n");
        // Write the CPU Matrix to text file for analysis
        char fileNameCPU[100];
        sprintf(fileNameCPU, "resultsIntsCPU.txt");

        printMatrixToFileInts(fileNameCPU, MCPU);
    }

    char fileName[FILENAME_MAX_LENGTH];
    sprintf(fileName, "Test/Addition_%s_Ints_Runtime_Matrix_Size_%dx%d.csv", kernelName, M3Rows, M3Cols);
    printMatrixToFileInts(fileName, M3);

    freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

    // Exit program
    return 0;
}