#ifndef MATRIX_CUH
#define MATRIX_CUH

struct Matrix
{
    int rows;
    int cols;
    float *data;
};

Matrix createMatrix(int rows, int cols);

void populateWithOnes(Matrix matrix);
void populateWithRandomFloats(Matrix matrix);
void printMatrixToFile(char *fileName, Matrix M);
bool compareMatrices(Matrix M1, Matrix M2);
void initializeMatricesAndMemory(Matrix &M1, Matrix &M2, Matrix &M3, int M1Rows, int M1Cols, int M2Rows, int M2Cols, int M3Rows, int M3Cols);
void allocateMemoryOnGPU(float *&device_M1, float *&device_M2, float *&device_M3, size_t memorySize1, size_t memorySize2, size_t memorySize3);
void copyMatricesToGPU(const Matrix &M1, const Matrix &M2, float *device_M1, float *device_M2, size_t memorySize1, size_t memorySize2);
void freeMemory(double *device_M1, double *device_M2, double *device_M3, Matrix &M1, Matrix &M2, Matrix &M3);

#endif