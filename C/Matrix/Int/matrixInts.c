#include "matrixInts.h"
#include <stdio.h>
#include <stdlib.h>

// Create a matrix
struct MatrixInts createMatrixInts(int rows, int cols) {
    struct MatrixInts matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = (int *)malloc(rows * cols * sizeof(int));
    
    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }
    
    return matrix;
}

// Set all indices in matrix to hold value 1
void populateWithOnesInts(MatrixInts matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1;
        }
    }
}

// Generate random integers on the CPU using srand
void populateWithRandomInts(MatrixInts matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = rand(); // rand() generates a random number based on the seed (CPU only)
        }
    }
}