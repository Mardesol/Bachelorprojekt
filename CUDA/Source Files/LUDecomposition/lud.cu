#include "ludKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"

const bool printDebugMessages = true;
const size_t FILENAME_MAX_LENGTH = 256;

const char *executeChosenKernel(int KernelNumToPerform, float *device_A, int ADim, Timer timer)
{
    dim3 blockDim(16,16);
    dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

    const char *kernelName;

    switch (KernelNumToPerform)
    {
    case 1:
        kernelName = "Sequential LUD (GPU)";
        beginTimer(timer);
        Sequential<<<gridDim, blockDim>>>(device_A, ADim);
        cudaDeviceSynchronize();
        endTimer(timer, "Sequential LUD (GPU)", printDebugMessages);
        break;
    case 2:
        kernelName = "Sequential LUD with pivoting (GPU)";
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
    MatrixF A;
    float *device_A;
    
    A = createMatrixFloats(ADim, ADim);
    populateWithRandomFloats(A);

    cudaMalloc((void **)&device_A, memorySize);
    cudaMemcpy(device_A, A.data, memorySize, cudaMemcpyHostToDevice);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

    // Execute Cuda Kernel
    const char *kernelName = executeChosenKernel(KernelNumToPerform, device_A, ADim, timer);
    cudaMemcpy(A.data, device_A, memorySize, cudaMemcpyDeviceToHost);

    // Setup a CPU comparison matrix
    //float** A_CPU_2D = MatrixF_to_twoDim(createMatrixFloats(ADim, ADim));
    //LUD_Sequential(A_CPU_2D, ADim);
    //MatrixF A_CPU_1D = twoDim_to_MatrixF(A_CPU_2D, ADim, ADim);

    MatrixF A_CPU = createMatrixFloats(ADim, ADim);
    populateWithRandomFloats(A_CPU);

    //LUD_Sequential(A_CPU.data, ADim);
    LUD_Sequential_Partial_Pivoting(A_CPU.data, ADim);

    // Validate result by comparing to CPU calculations
    bool valid = compareMatricesFloats(A, A_CPU);
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

        printMatrixToFileFloats(fileNameCPU, A_CPU);
    }

    char fileName[FILENAME_MAX_LENGTH];
    sprintf(fileName, "Test/LUD_%s_Runtime_Matrix_Size_%dx%d.csv", kernelName, ADim, ADim);
    printMatrixToFileFloats(fileName, A);

    cudaFree(device_A);
    free(A.data);
    free(A_CPU.data);

    // Exit program
    return 0;
}

    //printf("Setup 2d matrix, %f \n", A_CPU_2D);
    //printf("CPU calculations done");
    //printf("Converted to 1D matrix, %f", A_CPU_1D);