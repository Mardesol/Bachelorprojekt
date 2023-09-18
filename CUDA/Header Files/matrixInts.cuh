#ifndef MATRIXINTS_CUH
#define MATRIXINTS_CUH

struct MatrixI {
    int rows;
    int cols;
    int* data;
};

MatrixI createMatrixI(int rows, int cols);

void populateWithOnesI(MatrixI matrix);
void populateWithRandomInts(MatrixI matrix);

#endif
