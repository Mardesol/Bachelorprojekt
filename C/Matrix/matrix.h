#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix
{
    int rows;
    int cols;
    float *data;
} Matrix;

Matrix create_C_Matrix(int rows, int cols);
void populateWithOnes(Matrix matrix);
void populateWithRandomFloats(Matrix matrix);

#endif