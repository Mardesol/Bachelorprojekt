#include "..\..\Header Files\matrixInts.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

// Create a matrix on the host
MatrixI createMatrixInts(int rows, int cols) {
    MatrixI matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (int*)malloc(rows * cols * sizeof(int));

    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1
void populateWithOnesInts(MatrixI matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1;
        }
    }
}


// Generate random integers on the CPU using srand
void populateWithRandomInts(MatrixI matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = rand(); // rand() generates a random number based on the seed (CPU only)
        }
    }
}

// Comparison for int data type
bool compareMatricesInt(int* M1, int* M2, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            if (M1[i * MCols + j] != M2[i * MCols + j]) {
                return false;  // Matrices do not match
            }
        }
    }
    return true;  // Matrices match
}
