#include "ludKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"

const bool printDebugMessages = true;
const size_t FILENAME_MAX_LENGTH = 256;



const char *executeChosenKernel(int KernelNumToPerform, float *device_A, float* A_CPU_Data, int ADim, Timer timer)
{
    dim3 blockDim(16,16);
    dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

    // #define TILE_SIZE 16
    // dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // dim3 gridDim((ADim + TILE_SIZE - 1) / TILE_SIZE, (ADim + TILE_SIZE - 1) / TILE_SIZE);
    // size_t sharedMemSize = TILE_SIZE * TILE_SIZE * sizeof(float);

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
        kernelName = "New_Sequential";
        LUD_Sequential(A_CPU_Data, ADim);
        beginTimer(timer);
        New_Sequential << <1, 1>> > (device_A, ADim);
        endTimer(timer, "New_Sequential", printDebugMessages);
        break;
    case 4:
        kernelName = "New_Sequential_With_Partial_Pivoting";
        LUD_Sequential_Partial_Pivoting(A_CPU_Data, ADim);
        beginTimer(timer);
        New_Sequential_With_Partial_Pivoting << <1, 1 >> > (device_A, ADim);
        endTimer(timer, "New_Sequential_With_Partial_Pivoting", printDebugMessages);
        break;
    case 5:
        kernelName = "Right_Looking_Parallel_LUD";
        LUD_Sequential(A_CPU_Data, ADim);
        beginTimer(timer);
        Right_Looking_Parallel_LUD(device_A, ADim, blockDim);
        endTimer(timer, "Right_Looking_Parallel_LUD", printDebugMessages);
        break;
    /*case 6:
        kernelName = "Right_Looking_Parallel_LUD_With_Partial_Pivoting";
        LUD_Sequential_Partial_Pivoting(A_CPU_Data, ADim);
        beginTimer(timer);
        Right_Looking_Parallel_LUD_With_Partial_Pivoting(device_A, ADim);
        endTimer(timer, "Right_Looking_Parallel_LUD_With_Partial_Pivoting", printDebugMessages);
        break;*/
    
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
    char fileNameInput[100];
    sprintf(fileNameInput, "input.txt");
    printMatrixToFile(fileNameInput, A);
    const char *kernelName = executeChosenKernel(KernelNumToPerform, device_A, A_CPU.data, ADim, timer);
    cudaMemcpy(A.data, device_A, memorySize, cudaMemcpyDeviceToHost);

    // Validate result by comparing to CPU calculations
    // Write the CPU Matrix to text file for analysis
    char fileNameDiff[100];
    sprintf(fileNameDiff, "Test/Differences_%d.txt", KernelNumToPerform);

    bool valid = compareAndPrintDifferences(A,A_CPU, fileNameDiff);
    if (valid)
    {
        printf("Matrix LUD results match!\n");
    }
    else
    {
        printf("Matrix LUD results do not match.\n");
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