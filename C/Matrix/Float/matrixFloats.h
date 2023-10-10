#ifndef MATRIXFLOATS_H
#define MATRIXFLOATS_H

// Define Matrix as a typedef of struct Matrix
typedef struct MatrixFloats
{
    int rows;
    int cols;
    float *data;
} MatrixFloats;

// Declare function prototypes
MatrixFloats createMatrixFloats(int rows, int cols);
void populateWithOnesFloats(MatrixFloats matrix);
void populateWithRandomFloats(MatrixFloats matrix);

#endif
