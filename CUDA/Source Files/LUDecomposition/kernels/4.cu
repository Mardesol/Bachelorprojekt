// LU Decomposition for a square block
__global__ void LU_Diagonal(float *A, int dsize, int ldA) {
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

// DTRSM operation
__global__ void DTRSM(float *A, float *B, int dsize, int tsize, int ldA, bool left) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (left) {
        for (int i = 0; i < dsize; ++i) {
            __syncthreads();
            if (tx == i && ty >= dsize) {
                B[ty * ldA + tx] /= A[tx * ldA + tx];
            }
            __syncthreads();
            if (tx > i && ty >= dsize && tx < dsize && ty < tsize) {
                B[ty * ldA + tx] -= A[ty * ldA + i] * B[i * ldA + tx];
            }
        }
    } else {
        // Implement right-side DTRSM (optional)
    }
}

// DGEMM operation
__global__ void DGEMM(float *A, float *B, float *C, int dsize, int tsize, int ldA) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (tx >= dsize || ty >= tsize) return;
    float sum = 0.0f;
    for (int k = 0; k < dsize; ++k) {
        sum += A[ty * ldA + k] * B[k * ldA + tx];
    }
    C[ty * ldA + tx] -= sum;
}

void right_looking_lu(float *device_A, int ADim) {
    int block_size = 8;  // You can set this according to your needs
    int block_count = (ADim + block_size - 1) / block_size;
    dim3 threadsPerBlock(block_size, block_size);

    for (int i = 0; i < block_count; ++i) {
        int dsize = min(block_size, ADim - i * block_size);
        float *block_ii = device_A + (i * ADim + i) * block_size;

        LU_Diagonal<<<1, threadsPerBlock>>>(block_ii, dsize, ADim);
        cudaDeviceSynchronize();

        if (i < block_count - 1) {
            float *block_cols = device_A + (i * ADim * block_size) + (i + 1) * block_size;
            DTRSM<<<1, threadsPerBlock>>>(block_ii, block_cols, dsize, ADim - (i + 1) * block_size, ADim, true);
            cudaDeviceSynchronize();

            for (int j = i + 1; j < block_count; ++j) {
                float *block_row = device_A + ((i + 1) * ADim * block_size) + i * block_size;
                float *block_trail = device_A + ((i + 1) * ADim * block_size) + (i + 1) * block_size;

                DGEMM<<<1, threadsPerBlock>>>(block_row, block_cols, block_trail, dsize, ADim - (i + 1) * block_size, ADim);
                cudaDeviceSynchronize();
            }
        }
    }
}


