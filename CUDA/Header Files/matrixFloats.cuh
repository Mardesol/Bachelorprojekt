#ifndef MATRIXF_CUH
#define MATRIXF_CUH

struct MatrixF {
    int rows;
    int cols;
    float* data;
};

MatrixF createMatrixF(int rows, int cols);

void populateWithOnesF(MatrixF matrix);
void populateWithRandomFloats(MatrixF matrix);

#endif