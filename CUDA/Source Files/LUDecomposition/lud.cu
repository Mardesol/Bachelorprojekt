#include "ludKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"

const bool printDebugMessages = true;
const size_t FILENAME_MAX_LENGTH = 256;

const char *executeChosenKernel(int KernelNumToPerform, float *device_A, float* A_CPU_Data, int ADim, Timer timer)
{
    dim3 blockDim(16,16);
    dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

    const char *kernelName;

    switch (KernelNumToPerform)
    {
    case 1:
        kernelName = "Sequential LUD (GPU)";
        LUD_Sequential(A_CPU_Data, ADim);
        beginTimer(timer);
        Sequential<<<gridDim, blockDim>>>(device_A, ADim);
        cudaDeviceSynchronize();
        endTimer(timer, "Sequential LUD (GPU)", printDebugMessages);
        break;
    case 2:
        kernelName = "Sequential LUD with pivoting (GPU)";
        LUD_Sequential_Partial_Pivoting(A_CPU_Data, ADim);
        beginTimer(timer);
        Sequential_Partial_Pivoting<<<1, 1>>>(device_A, ADim);
        cudaDeviceSynchronize();
        endTimer(timer, "Sequential LUD with pivoting (GPU)", printDebugMessages);
        break;
    case 3:
        kernelName = "LUD_Block";
        beginTimer(timer);
        LUD_Block<<<gridDim, blockDim>>>(device_A, ADim);
        cudaDeviceSynchronize();
        endTimer(timer, "LUD_Block", printDebugMessages);
        break;
    case 4:
        kernelName = "LUD_Block_similar";
        beginTimer(timer);
        LUD_Block_Similar<<<gridDim, blockDim>>>(device_A, ADim);
        cudaDeviceSynchronize();
        endTimer(timer, "LUD_Block_Similar", printDebugMessages);
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
    int ADim = atoi(argv[2]); 

    size_t memorySize = ADim * ADim * sizeof(float);

    Timer timer = createTimer();

    beginTimer(timer);
    Matrix A;
    float *device_A;
 
    A = createMatrix(ADim, ADim);
    populateWithRandomFloats(A);

    Matrix A_CPU = createMatrix(ADim, ADim);
    populateWithRandomFloats(A_CPU);

    cudaMalloc((void **)&device_A, memorySize);
    cudaMemcpy(device_A, A.data, memorySize, cudaMemcpyHostToDevice);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    // Execute Cuda Kernel
    const char *kernelName = executeChosenKernel(KernelNumToPerform, device_A, A_CPU.data, ADim, timer);
    cudaMemcpy(A.data, device_A, memorySize, cudaMemcpyDeviceToHost);

    // Validate result by comparing to CPU calculations
    bool valid = compareMatrices(A, A_CPU);
    if (valid)
    {
        printf("Matrix LUD results match!\n");
    }
    else
    {
        printf("Matrix LUD results do not match.\n");
        // Write the CPU Matrix to text file for analysis
        char fileNameCPU[100];
        sprintf(fileNameCPU, "Test/resultsCPU.txt");

        printMatrixToFile(fileNameCPU, A_CPU);
    }

    char fileName[FILENAME_MAX_LENGTH];
    sprintf(fileName, "Test/LUD_%s_Runtime_Matrix_Size_%dx%d.csv", kernelName, ADim, ADim);
    printMatrixToFile(fileName, A);

    cudaFree(device_A);
    free(A.data);
    free(A_CPU.data);

    // Exit program
    return 0;
}