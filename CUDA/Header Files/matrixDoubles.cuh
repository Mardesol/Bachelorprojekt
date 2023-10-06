#ifndef MATRIXDOUBLES_CUH
#define MATRIXDOUBLES_CUH

struct MatrixD {
    int rows;
    int cols;
    double* data;
};

MatrixD createMatrixD(int rows, int cols);

void populateWithOnesD(MatrixD matrix);
void populateWithRandomDoubles(MatrixD matrix);

bool compareMatricesDouble(double* M1, double* M2, int MRows, int MCols);

#endif
