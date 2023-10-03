#ifndef MATRIXINTS_H
#define MATRIXINTS_H

// Define Matrix as a typedef of struct Matrix
typedef struct MatrixInts {
    int rows;
    int cols;
    int *data;
} MatrixInts;

// Declare function prototypes
MatrixInts createMatrixInts(int rows, int cols);
void populateWithOnesInts(MatrixInts matrix);
void populateWithRandomInts(MatrixInts matrix);

#endif
