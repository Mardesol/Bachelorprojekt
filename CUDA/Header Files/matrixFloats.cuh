#ifndef MATRIXFLOATS_CUH
#define MATRIXFLOATS_CUH

struct MatrixF
{
    int rows;
    int cols;
    float *data;
};

MatrixF createMatrixFloats(int rows, int cols);

void populateWithOnesFloats(MatrixF matrix);
void populateWithRandomFloats(MatrixF matrix);
void printMatrixToFileFloats(char *fileName, MatrixF M);
bool compareMatricesFloats(MatrixF M1, MatrixF M2);
void initializeMatricesAndMemory(MatrixF &M1, MatrixF &M2, MatrixF &M3);
void allocateMemoryOnGPU(double *&device_M1, double *&device_M2, double *&device_M3);
void copyMatricesToGPU(const MatrixF &M1, const MatrixF &M2, double *device_M1, double *device_M2);
void freeMemory(double *device_M1, double *device_M2, double *device_M3, MatrixF &M1, MatrixF &M2, MatrixF &M3);

#endif