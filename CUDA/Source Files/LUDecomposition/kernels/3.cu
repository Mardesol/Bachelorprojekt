// Kernel for LU decomposition on diagonal block A_ii
__global__ void LU_Diagonal(float* A, int i, int b, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global row and column
    int row = i + ty;
    int col = i + tx;

    __shared__ float tile[32][32];  // Assume block size is 32x32

    // Load tile into shared memory
    if (row < N && col < N) {
        tile[ty][tx] = A[row * N + col];
    }

    __syncthreads();

    for (int k = 0; k < b; ++k) {
        // Update diagonal and below
        if (tx == k) {
            for (int j = k + 1; j < b; ++j) {
                tile[j][k] /= tile[k][k];
            }
        }

        __syncthreads();

        // Update the rest of the block
        for (int j = k + 1; j < b; ++j) {
            for (int l = k + 1; l < b; ++l) {
                tile[j][l] -= tile[j][k] * tile[k][l];
            }
        }

        __syncthreads();
    }

    // Write back to global memory
    if (row < N && col < N) {
        A[row * N + col] = tile[ty][tx];
    }
}

__global__ void Update_OffDiagonalBlocks(float* A, int i, int b, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = i + ty;
    int col = i + b + tx;

    __shared__ float tile_Aii[32][32];  // For A_ii block
    __shared__ float tile_Aij[32][32];  // For A_ij block
    __shared__ float tile_Aji[32][32];  // For A_ji block

    // Load A_ii into shared memory
    if (row < N && tx < b) {
        tile_Aii[ty][tx] = A[row * N + i + tx];
    }

    // Load A_ij and A_ji into shared memory
    if (row < N && col < N) {
        tile_Aij[ty][tx] = A[row * N + col];
        tile_Aji[tx][ty] = A[col * N + row];
    }

    __syncthreads();

    // Update A_ij = A_ij - A_ik * A_kj
    // Update A_ji = A_ji - A_jk * A_ki
    for (int k = 0; k < b; ++k) {
        float value_Aii = tile_Aii[ty][k];
        float value_Aji = tile_Aji[tx][k];
        for (int j = 0; j < b; ++j) {
            tile_Aij[ty][j] -= value_Aii * tile_Aji[j][k];
            tile_Aji[j][ty] -= value_Aji * tile_Aii[j][k];
        }
    }

    __syncthreads();

    // Write back to global memory
    if (row < N && col < N) {
        A[row * N + col] = tile_Aij[ty][tx];
        A[col * N + row] = tile_Aji[tx][ty];
    }
}

// Kernel to update the Schur Complement A_kj and A_ik
__global__ void Update_SchurComplement(float* A, int i, int j, int k, int b, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = i + b + ty;
    int col = j + b + tx;

    __shared__ float tile_Aik[32][32];  // For A_ik block
    __shared__ float tile_Akj[32][32];  // For A_kj block

    // Load A_ik and A_kj into shared memory
    if (row < N && tx < b) {
        tile_Aik[ty][tx] = A[row * N + i + tx];
    }

    if (ty < b && col < N) {
        tile_Akj[ty][tx] = A[(i + ty) * N + col];
    }

    __syncthreads();

    // Update A_kj = A_kj - A_ik * A_kj
    for (int l = 0; l < b; ++l) {
        float value = 0.0f;
        for (int m = 0; m < b; ++m) {
            value += tile_Aik[ty][m] * tile_Akj[m][tx];
        }
        A[row * N + col] -= value;
    }
}

void LUD_Blocking(float* device_A, int ADim) {
    const int b = 16;                   // Block size, assuming 16x16 blocks
    const dim3 threadsPerBlock(16, 16); 
    dim3 gridDim((ADim + threadsPerBlock.x - 1) / threadsPerBlock.x, (ADim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i = 0; i < ADim; i += b) {
        // Step 1: LU Decompose the diagonal block A_ii
        LU_Diagonal<<<1, threadsPerBlock>>>(device_A, i, b, ADim);
        cudaDeviceSynchronize();

        // Step 2: Update the off-diagonal blocks in row i and column i
        Update_OffDiagonalBlocks<<<gridDim, threadsPerBlock>>>(device_A, i, b, ADim);
        cudaDeviceSynchronize();

        // Step 3: Update the remaining blocks (Schur Complement blocks)
        for (int j = i + b; j < ADim; j += b) {
            for (int k = i + b; k < ADim; k += b) {
                Update_SchurComplement<<<1, threadsPerBlock>>>(device_A, i, j, k, b, ADim);
            }
        }
        cudaDeviceSynchronize();
    }
}