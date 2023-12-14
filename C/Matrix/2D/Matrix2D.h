#ifndef MATRIX2D_H
#define MATRIX2D_H

typedef struct Matrix2D
{
    int rows;
    int cols;
    float **data;
} Matrix2D;

Matrix2D createMatrix2D(int rows, int cols);
void populateWithOnes(Matrix2D matrix);
void populateWithRandomFloats(Matrix2D matrix);
void printMatrix2DToFile(char *fileName, Matrix2D matrix);

#endif