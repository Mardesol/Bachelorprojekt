// Block A
__global__ void LU_Diagonal(float* A, int dsize, int ldA) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int k = 0; k < dsize; ++k) {
        __syncthreads();
        if (tx > k && tx < dsize) {
            A[tx * ldA + k] /= A[k * ldA + k];
        }
        __syncthreads();
        if (tx > k && ty > k && tx < dsize && ty < dsize) {
            A[ty * ldA + tx] -= A[ty * ldA + k] * A[k * ldA + tx];
        }
        __syncthreads();
    }
}

// Block B
__global__ void ForwardSubstitution(float* L, float* B, int dsize, int ldB, int ldA, int i, int blockSize) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    if (tx < blockSize && ty < dsize) {
        float sum = B[tx * ldB + ty];
        for (int k = 0; k < tx; k++) {
            sum -= L[tx * ldA + k] * B[k * ldB + ty];
            B[tx * ldB + ty] = sum;
            __syncthreads();
        }
    }
}


//// Block C
//__global__ void Schur(float* C, const float* A, const float* B, int rowsC, int colsC, int rowsA, int colsB, int blockSize) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (row < rowsC && col < colsC) {
//        float sum = 0.0f;
//        for (int k = blockSize - 1; k >= col; k--) {
//            float A_val;
//            if (col == k) { 
//                A_val = 1.0f; 
//            } else if (col > k) {
//                A_val = 0.0f;
//            } else {
//                A_val = A[row * rowsA + k * rowsA];
//            }
//            printf("Im on index %d, %d, k: %d, calculating sum: %f += A_val %f * %f \n", row, col, k, sum, A_val, B[col * blockSize + col * colsB]);
//            //printf("B is calculated: %d + %d * %d + %d  = %d \n", colsB, col, rowsA, k, colsB + col * rowsA + k);
//            sum += A_val * B[col * blockSize + col * colsB];
//            __syncthreads();
//        }
//        printf("Im on index %d, %d, calculating %f -= sum: %f \n", row, col, C[row * rowsA + col], sum);
//        C[row * rowsA + col] -= sum;
//    }
//}

// Block C
__global__ void Schur(float* C, const float* A, const float* B, int rowsC, int colsC, int rowsA, int colsB, int blockSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsC && col < colsC) {
        float sum = 0.0f;
        for (int k = 0; k < colsC; ++k) {
            float A_val;
            if (k == row) {
                A_val = 1.0f;
            }
            else if (k > col){
                A_val = 0.0f;
            }
            else {
                A_val = A[col * blockSize + k];
            }
            printf("K is %d, I'm in index %d, %d, and calculating %f += %f * %f \n", k, row, col, sum, A_val, C[col]);
            sum += A_val * C[col];
        }
        C[col] = sum;
    }
}



// Block D
__global__ void LUD_Update_D(float* L, float* U, float* D, float* D_out, int ldA, int dsizeL, int dsizeU) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tx >= dsizeL || ty >= dsizeU) return;

    float sum = 0.0f;
    for (int k = 0; k < dsizeL; ++k) {  // Iterate over the common dimension
        sum += L[ty * ldA + k] * U[k * ldA + tx];  // L*U multiplication
    }
    __syncthreads();
    // Update D block as part of LU decomposition
    D[ty * ldA + tx] -= sum;
    __syncthreads();
    // Write the updated D block to D_out for debugging
    D_out[ty * ldA + tx] = D[ty * ldA + tx];
}

void right_looking_lu(float* device_A, int ADim) {
    int block_size = 4;  // Adjust as needed
    int block_count = (ADim + block_size - 1) / block_size;

    for (int i = 0; i < block_count; ++i) {
        int currentBlockSize = min(block_size, ADim - i * block_size);
        float* block_ii = device_A + (i * ADim + i) * block_size;

        dim3 diagBlock(currentBlockSize, currentBlockSize);
        LU_Diagonal << <1, diagBlock >> > (block_ii, currentBlockSize, ADim);
        cudaDeviceSynchronize();

        if (i < block_count - 1) {
            float* block_ij = device_A + (i * ADim + (i + 1)) * block_size;
            float* block_ji = device_A + ((i + 1) * ADim + i) * block_size;
            float* block_jj = device_A + ((i + 1) * ADim + (i + 1)) * block_size;
            int tsize = ADim - (i + 1) * block_size;  // Size of the remaining matrix columns
            int dsize = ADim - (i + 1) * block_size;  // Size of the remaining matrix rows

            // ForwardSubstitution kernel invocation for block_ij (B block)
            int numRowsB = block_size;
            int numColsB = tsize;
            dim3 fsBlock(block_size);
            dim3 fsGrid((numColsB + block_size - 1) / block_size, (ADim + block_size -1) / block_size);
            ForwardSubstitution << <fsGrid, fsBlock >> > (block_ii, block_ij, dsize, ADim, ADim, i, block_size);
            cudaDeviceSynchronize();

            // Schur kernel invocation for block_ji (C block)
            int numRowsC = dsize;
            int numColsC = block_size;
            dim3 B_Block(block_size);
            dim3 B_Grid((ADim + block_size - 1) / block_size, (numRowsC + block_size - 1) / block_size);
            Schur << <B_Grid, B_Block >> > (block_ji, block_ii, block_ij, numRowsC, numColsC, ADim, numColsB, block_size);
            cudaDeviceSynchronize();

            // DGEMM kernel update for the D block
            //int numRowsD= (dsize + block_size - 1) / block_size;
            //int numColsD = (tsize + block_size - 1) / block_size;
            //dim3 grid(gridCols, gridRows);
            //LUD_Update_D << <grid, threadsPerBlock >> > (block_ji, block_ij, block_jj, device_D_out + ((i + 1) * ADim + (i + 1)) * block_size, ADim, dsize, tsize);
            //cudaDeviceSynchronize();
        }
    }
}