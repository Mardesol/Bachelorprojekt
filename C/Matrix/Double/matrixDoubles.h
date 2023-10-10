#ifndef MATRIXFLOATS_H
#define MATRIXFLOATS_H

// Define Matrix as a typedef of struct Matrix
typedef struct MatrixDoubles
{
    int rows;
    int cols;
    double *data;
} MatrixDoubles;

// Declare function prototypes
MatrixDoubles createMatrixFloats(int rows, int cols);
void populateWithOnesFloats(MatrixDoubles matrix);
void populateWithRandomFloats(MatrixDoubles matrix);

#endif
