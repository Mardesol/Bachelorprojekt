#include "..\..\Header Files\matrixFloats.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

// Create a matrix on the host
MatrixF createMatrixF(int rows, int cols) {
    MatrixF matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (float*)malloc(rows * cols * sizeof(float));

    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1.0f
void populateWithOnesF(MatrixF matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1.0f; // Change to float
        }
    }
}

// Generate random floats on the CPU using srand
void populateWithRandomFloats(MatrixF matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Generate random float between 0 and 1
        }
    }
}

// Comparison for float data type
bool compareMatricesFloat(float* M1, float* M2, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            if (M1[i * MCols + j] != M2[i * MCols + j]) {
                return false;  // Matrices do not match
            }
        }
    }
    return true;  // Matrices match
}
