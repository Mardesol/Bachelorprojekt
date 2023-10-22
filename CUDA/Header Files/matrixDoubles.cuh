#ifndef MATRIXDOUBLES_CUH
#define MATRIXDOUBLES_CUH

struct MatrixD
{
    int rows;
    int cols;
    double *data;
};

MatrixD createMatrixDoubles(int rows, int cols);

void populateWithOnesDoubles(MatrixD matrix);
void populateWithRandomDoubles(MatrixD matrix);
void printMatrixToFileDoubles(char *fileName, MatrixD M);
bool compareMatricesDoubles(MatrixD M1, MatrixD M2);
void initializeMatricesAndMemory(MatrixD &M1, MatrixD &M2, MatrixD &M3, int M1Rows, int M1Cols, int M2Rows, int M2Cols, int M3Rows, int M3Cols);
void allocateMemoryOnGPU(double *&device_M1, double *&device_M2, double *&device_M3, size_t memorySize1, size_t memorySize2, size_t memorySize3);
void copyMatricesToGPU(const MatrixD &M1, const MatrixD &M2, double *device_M1, double *device_M2, size_t memorySize1, size_t memorySize2);
void freeMemory(double *device_M1, double *device_M2, double *device_M3, MatrixD &M1, MatrixD &M2, MatrixD &M3);

#endif