#ifndef MATRIXOPERATIONSCPU_CUH
#define MATRIXOPERATIONSCPU_CUH

#include "matrix.cuh"

void addition(Matrix M1, Matrix M2, Matrix M3);
void multiplication(Matrix M1, Matrix M2, Matrix M3);
bool compareMatrices(Matrix M1, Matrix M2);

void LUD_Sequential(float *A, int n);
void LUD_Sequential_Partial_Pivoting(float** A, int n);

#endif