//Sequential kernels

__global__ void New_Sequential(float* A, int n) {
    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {
        
        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
        }
        // Compute U elements (upper triangular part)
        for (int j = i + 1; j < n; j++) {
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] = A[j * n + k] - A[i * n + k] * A[j * n + i];
            }
        }
    }
}

__global__ void New_Sequential_With_Partial_Pivoting(float* A, int n) {
    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        // Find pivot row
        int pivotRow = i;
        float maxVal = fabsf(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabsf(A[p * n + i]) > maxVal) {
                maxVal = fabsf(A[p * n + i]);
                pivotRow = p;
            }
        }

        // Swap rows if needed
        if (pivotRow != i) {
            for (int j = 0; j < n; j++) {
                float temp = A[i * n + j];
                A[i * n + j] = A[pivotRow * n + j];
                A[pivotRow * n + j] = temp;
            }
        }

        __syncthreads();

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
        }
        // Compute U elements (upper triangular part)
        for (int j = i + 1; j < n; j++) {
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] = A[j * n + k] - A[i * n + k] * A[j * n + i];
            }
        }
    }
}


//Parallel kernels and methods

__global__ void ComputeL(float* A, int n, int i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    if (j < n) {
        printf("calculating index %d \n", j * n + i);
        A[j * n + i] = A[j * n + i] / A[i * n + i];
    }
}

__global__ void ComputeU(float* A, int n, int i) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int j = index / (n - i - 1) + i + 1;
    int k = index % (n - i - 1) + i + 1;
    if (j < n && k < n) {
        printf("calculating index %d \n", j * n + k);
        A[j * n + k] = A[j * n + k] - A[i * n + k] * A[j * n + i];
    }
}

void findPivotAndSwapRows(float* A, int n, int i) {
    // Find pivot row
    int pivotRow = i;
    float maxVal = fabsf(A[i * n + i]);

    for (int p = i + 1; p < n; p++) {
        if (fabsf(A[p * n + i]) > maxVal) {
            maxVal = fabsf(A[p * n + i]);
            pivotRow = p;
        }
    }

    // Swap rows if needed
    if (pivotRow != i) {
        for (int j = 0; j < n; j++) {
            float temp = A[i * n + j];
            A[i * n + j] = A[pivotRow * n + j];
            A[pivotRow * n + j] = temp;
        }
    }
}

__global__ void PivotAndSwapRowsKernel(float* A, int n, int i, int* pivotRow) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0) {
        *pivotRow = i;
        float maxVal = fabsf(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabsf(A[p * n + i]) > maxVal) {
                maxVal = fabsf(A[p * n + i]);
                *pivotRow = p;
            }
        }
    }
    __syncthreads();

    int currentPivotRow = *pivotRow;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float temp = A[i * n + j];
        A[i * n + j] = A[currentPivotRow * n + j];
        A[currentPivotRow * n + j] = temp;
    }
}


void Right_Looking_Parallel_LUD(float* A, int n) {

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {
        printf("loop %d \n", i);
        // Set up kernel launch parameters for L computation
        int blockSizeL = 32;
        int numBlocksL = (n - i - 1 + blockSizeL - 1) / blockSizeL;
        ComputeL << <numBlocksL, blockSizeL >> > (A, n, i);
        cudaDeviceSynchronize();


        // Set up kernel launch parameters for U computation
        int totalElementsU = (n - i - 1) * (n - i - 1);
        int blockSizeU = 32;
        int numBlocksU = (totalElementsU + blockSizeU - 1) / blockSizeU;
        ComputeU << <numBlocksU, blockSizeU >> > (A, n, i);
        cudaDeviceSynchronize();
        
    }
}

void Right_Looking_Parallel_LUD_With_Partial_Pivoting(float* A, int n) {
    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        // Perform partial pivoting
        int* device_pivotRow;  // Allocate memory for pivotRow on the GPU
        cudaMalloc((void**)&device_pivotRow, sizeof(int));
        PivotAndSwapRowsKernel << <1, 32 >> > (A, n, i, device_pivotRow);
        cudaDeviceSynchronize();

        // Get the pivot row index from the GPU
        int pivotRow;
        cudaMemcpy(&pivotRow, device_pivotRow, sizeof(int), cudaMemcpyDeviceToHost);

        // Set up kernel launch parameters for L computation
        int blockSizeL = 32;
        int numBlocksL = (n - i - 1 + blockSizeL - 1) / blockSizeL;
        ComputeL << <numBlocksL, blockSizeL >> > (A, n, i);
        cudaDeviceSynchronize();

        // Set up kernel launch parameters for U computation
        int totalElementsU = (n - i - 1) * (n - i - 1);
        int blockSizeU = 32;
        int numBlocksU = (totalElementsU + blockSizeU - 1) / blockSizeU;
        ComputeU << <numBlocksU, blockSizeU >> > (A, n, i);
        cudaDeviceSynchronize();

    }
}