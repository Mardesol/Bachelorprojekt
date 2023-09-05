#ifndef MATRIX_CUH
#define MATRIX_CUH

struct Matrix {
    int rows;
    int cols;
    int* data;
};

Matrix createMatrix(int rows, int cols);

void populateWithOnes(Matrix matrix);
void populateWithRandomInts(Matrix matrix);

#endif
