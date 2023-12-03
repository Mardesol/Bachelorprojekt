#include "..\Timer\timer.cu"
#include "..\Matrix\matrix.cu"

// CUDA kernel to add two matrices in parallel, utilizing both thread and block level parallelism
__global__ void Parallel(float *M1, float *M2, float *M3, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * rows + col;
        M3[index] = M1[index] + M2[index];
    }
}