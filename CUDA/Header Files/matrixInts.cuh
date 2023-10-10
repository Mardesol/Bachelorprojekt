#ifndef MATRIXINTS_CUH
#define MATRIXINTS_CUH

struct MatrixI
{
    int rows;
    int cols;
    int *data;
};

MatrixI createMatrixInts(int rows, int cols);

void populateWithOnesInts(MatrixI matrix);
void populateWithRandomInts(MatrixI matrix);
void printMatrixToFileInts(char *fileName, MatrixI M);
bool compareMatricesInts(MatrixI M1, MatrixI M2);
void initializeMatricesAndMemory(MatrixI &M1, MatrixI &M2, MatrixI &M3);
void allocateMemoryOnGPU(double *&device_M1, double *&device_M2, double *&device_M3);
void copyMatricesToGPU(const MatrixI &M1, const MatrixI &M2, double *device_M1, double *device_M2);
void freeMemory(double *device_M1, double *device_M2, double *device_M3, MatrixI &M1, MatrixI &M2, MatrixI &M3);

#endif