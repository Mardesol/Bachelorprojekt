#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

// Create a matrix
struct Matrix createMatrix(int rows, int cols) {
    struct Matrix matrix;
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
void populateWithOnes(struct Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i * matrix->cols + j] = 1;
        }
    }
}

