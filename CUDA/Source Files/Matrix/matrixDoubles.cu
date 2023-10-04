#include "..\..\Header Files\matrixDoubles.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

// Create a matrix on the host
MatrixD createMatrixDoubles(int rows, int cols) {
    MatrixD matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (double*)malloc(rows * cols * sizeof(double));

    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1.0
void populateWithOnesDoubles(MatrixD matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1.0;
        }
    }
}

// Generate random double values on the CPU
void populateWithRandomDoubles(MatrixD matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = (double)rand() / RAND_MAX; // Generates a random double between 0 and 1
        }
    }
}
