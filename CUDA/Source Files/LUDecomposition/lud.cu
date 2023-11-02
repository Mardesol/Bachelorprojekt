#include "ludKernels.cu"
#include "..\Matrix\matrixOperationsCPU.cu"
#include <cublas_v2.h>

const bool printDebugMessages = true;
const size_t FILENAME_MAX_LENGTH = 256;

void LUD_cuSolver(float *device_A, int ADim, Timer timer) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int *d_pivot, *d_info;
    cudaMalloc((void **)&d_pivot, ADim * sizeof(int));
    cudaMalloc((void **)&d_info, sizeof(int));

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, ADim, ADim, device_A, ADim, &lwork);
    float *work;
    cudaMalloc((void **)&work, lwork * sizeof(float));

    beginTimer(timer);
    cusolverDnSgetrf(handle, ADim, ADim, device_A, ADim, work, NULL, d_info);
    cudaDeviceSynchronize();
    endTimer(timer, "cuSolver LUD (GPU)", printDebugMessages);

    // Clean up
    cudaFree(d_pivot);
    cudaFree(d_info);
    cudaFree(work);
    cusolverDnDestroy(handle);
}

void LUD_cuBLAS_Single(float *device_A, int ADim, Timer timer) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int *d_info;
    cudaMalloc((void **)&d_info, sizeof(int));

    // Wrap the pointer to device_A in an array for batched processing with batch size 1
    float *device_A_array[1] = {device_A};

    beginTimer(timer);
    cublasSgetrfBatched(handle, ADim, device_A_array, ADim, NULL, d_info, 1);
    cudaDeviceSynchronize();
    endTimer(timer, "cuBLAS LUD Single (GPU)", printDebugMessages);

    // Clean up
    cudaFree(d_info);
    cublasDestroy(handle);
}

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
        kernelName = "cuSolver LUD (GPU)";
        LUD_Sequential(A_CPU_Data, ADim);
        LUD_cuSolver(device_A, ADim, timer);
        break;
    case 4:
        kernelName = "cuBLAS LUD Single (GPU)";
        LUD_Sequential(A_CPU_Data, ADim);
        LUD_cuBLAS_Single(device_A, ADim, timer);
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