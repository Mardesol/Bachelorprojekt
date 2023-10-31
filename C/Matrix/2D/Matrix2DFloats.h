#ifndef MATRIX2DFLOATS_H
#define MATRIX2DFLOATS_H

// Define Matrix as a typedef of struct Matrix
typedef struct Matrix2DFloats
{
    int rows;
    int cols;
    float **data;
} Matrix2DFloats;

// Declare function prototypes
Matrix2DFloats createMatrix2DFloats(int rows, int cols);
void populateWithOnesFloats(Matrix2DFloats matrix);
void populateWithRandomFloats(Matrix2DFloats matrix);
void printMatrix2DToFile(char *fileName, Matrix2DFloats matrix);

#endif