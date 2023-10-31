#ifndef MATRIX2DFLOATS_H
#define MATRIX2DFLOATS_H

// Define Matrix as a typedef of struct Matrix
typedef struct Matrix2D
{
    int rows;
    int cols;
    float **data;
} Matrix2D;

// Declare function prototypes
Matrix2D createMatrix2D(int rows, int cols);
void populateWithOnes(Matrix2D matrix);
void populateWithRandomFloats(Matrix2D matrix);
void printMatrix2DToFile(char *fileName, Matrix2D matrix);

#endif