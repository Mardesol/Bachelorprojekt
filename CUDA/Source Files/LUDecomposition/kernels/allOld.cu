void Parallel(float* A, int n, dim3 blockDim) {

    // Loop over each row - Must be done 1 at the time
    for (int i = 0; i < n; i++) {

        //Dimensions of the submatrix below/to the right of element (i,i)
        int subMatrixDim = n - i - 1;

        // Calculates the L values for row j
        dim3 blockDimColumn(1, blockDim.y);
        dim3 gridDimColumn(1, (subMatrixDim + blockDim.x - 1) / blockDim.x);
        ComputeLowerColumn << <gridDimColumn, blockDimColumn >> > (A, n, i);
        cudaDeviceSynchronize();

        dim3 gridDimSubmatrix((subMatrixDim + blockDim.x - 1) / blockDim.x, (subMatrixDim + blockDim.y - 1) / blockDim.y);
        UpdateSubmatrix << <gridDimSubmatrix, blockDim >> > (A, n, i);
        cudaDeviceSynchronize();

    }
}

__global__ void Sequential(float* A, int n) {
    // Loop over each row
    for (int i = 0; i < n; i++) {

        // Compute U elements (upper triangular part)
        for (int j = i; j < n; j++) {

            float sum = A[i * n + j];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        // Compute L elements (lower triangular part)
        for (int j = i + 1; j < n; j++) {

            float sum = A[j * n + i];
            // Subtract the lower * upper products from sum
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            // Divide by the diagonal element
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}

__global__ void Sequential_Partial_Pivoting(float* A, int n) {
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

        // Perform LUD
        for (int j = i; j < n; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        for (int j = i + 1; j < n; j++) {
            float sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}

__global__ void PivotAndSwapParallel(float* A, int* pivotIndices, int n, int i) {
    // Find the pivot: maximum element in the current column
    int maxIndex = i;
    float maxValue = abs(A[i * n + i]);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //if (tid == 0) {
    for (int row = i + 1; row < n; ++row) {
        float value = abs(A[row * n + i]);
        if (value > maxValue) {
            maxIndex = row;
            maxValue = value;
        }
    }
    //}

    pivotIndices[i] = maxIndex;
    __syncthreads();

    // Swap rows if necessary
    for (int col = tid; col < n; col += blockDim.x * gridDim.x) {
        float temp = A[i * n + col];
        A[i * n + col] = A[maxIndex * n + col];
        A[maxIndex * n + col] = temp;
    }
}