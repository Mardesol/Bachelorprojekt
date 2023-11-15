#ifndef MATRIX_H
#define MATRIX_H

// Define Matrix as a typedef of struct Matrix
typedef struct Matrix
{
    int rows;
    int cols;
    float *data;
} Matrix;

// Declare function prototypes
Matrix create_C_Matrix(int rows, int cols);
void populateWithOnes(Matrix matrix);
void populateWithRandomFloats(Matrix matrix);

#endif