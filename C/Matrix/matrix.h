#ifndef MATRIX_H
#define MATRIX_H

// Define Matrix as a typedef of struct Matrix
typedef struct Matrix {
    int rows;
    int cols;
    int *data;
} Matrix;

// Declare function prototypes
Matrix createMatrix(int rows, int cols);
void populateWithOnes(Matrix matrix);
void populateWithRandomInts(Matrix matrix);

#endif
