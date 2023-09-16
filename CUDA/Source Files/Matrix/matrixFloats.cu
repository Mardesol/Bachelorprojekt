#include "..\..\Header Files\matrixFloats.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

// Create a matrix on the host
MatrixF createMatrixFloats(int rows, int cols) {
    MatrixFloats matrix;
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
void populateWithOnesFloats(MatrixFloats matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1.0f; // Change to float
        }
    }
}

// Generate random floats on the CPU using srand
void populateWithRandomFloats(MatrixFloats matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Generate random float between 0 and 1
        }
    }
}
