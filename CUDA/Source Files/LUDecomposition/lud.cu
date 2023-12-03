#include "ludKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"

const bool printDebugMessages = true;
const size_t FILENAME_MAX_LENGTH = 256;
int* hostPivotIndices;


const char *executeChosenKernel(int KernelNumToPerform, float *device_A, float* A_CPU_Data, int ADim, Timer timer)
{
    dim3 blockDim(32,32);
    dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

    // #define TILE_SIZE 16
    // dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // dim3 gridDim((ADim + TILE_SIZE - 1) / TILE_SIZE, (ADim + TILE_SIZE - 1) / TILE_SIZE);
    // size_t sharedMemSize = TILE_SIZE * TILE_SIZE * sizeof(float);

    const char *kernelName;

    switch (KernelNumToPerform)
    {
    case 1:
        kernelName = "New_Sequential";
        LUD_Sequential(A_CPU_Data, ADim);
        beginTimer(timer);
        New_Sequential << <1, 1>> > (device_A, ADim);
        endTimer(timer, "New_Sequential", printDebugMessages);
        break;
    case 2:
        kernelName = "New_Sequential_With_Partial_Pivoting";
        LUD_Sequential_Partial_Pivoting(A_CPU_Data, ADim);
        beginTimer(timer);
        New_Sequential_With_Partial_Pivoting << <1, 1 >> > (device_A, ADim);
        endTimer(timer, "New_Sequential_With_Partial_Pivoting", printDebugMessages);
        break;
    case 3:
        kernelName = "Parallel_Partial_Pivot";
        beginTimer(timer);
        hostPivotIndices = Parallel_Pivoted(device_A, ADim, blockDim);
        endTimer(timer, "Parallel_Pivoted", printDebugMessages);
        break;
    case 4:
        kernelName = "SharedMemory_Pivoted";
        beginTimer(timer);
        hostPivotIndices = SharedMemory_Pivoted(device_A, ADim, blockDim);
        endTimer(timer, "SharedMemory_Pivoted", printDebugMessages);
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
    char fileNameInput[100];
    sprintf(fileNameInput, "input.txt");
    printMatrixToFile(fileNameInput, A);
    const char *kernelName = executeChosenKernel(KernelNumToPerform, device_A, A_CPU.data, ADim, timer);
    cudaMemcpy(A.data, device_A, memorySize, cudaMemcpyDeviceToHost);

    //Split the result into a L and U matrix
    Matrix L = createMatrix(ADim, ADim);
    Matrix U = createMatrix(ADim, ADim);
    printMatrixToFile("Original.txt", A);
    separateLU(A.data, L.data, U.data, ADim);
    printMatrixToFile("Lower.txt", L);
    printMatrixToFile("Upper.txt", U);

    //Multiply L and U for correctness check
    Matrix LUProduct = createMatrix(ADim, ADim);
    multiplication(L, U, LUProduct);
    printMatrixToFile("Product.txt", LUProduct);

    //Apply pivoting te the reconstructed matrix
    applyPivoting(LUProduct.data, hostPivotIndices, ADim);


    // Validate result by comparing to the input matrix
    char fileNameDiff[100];
    sprintf(fileNameDiff, "Test/Differences_%d_%dx%d.txt", KernelNumToPerform, ADim, ADim);

    bool valid = compareAndPrintDifferences(A_CPU, LUProduct, fileNameDiff);

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
    printMatrixToFile(fileName, LUProduct);

    cudaFree(device_A);
    free(A.data);
    free(A_CPU.data);

    // Exit program
    return 0;
}