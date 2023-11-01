#include "matrixDoubles.cuh"
#include "matrixFloats.cuh"
#include "matrixInts.cuh"

#ifndef MATRIXOPERATIONSCPU_CUH
#define MATRIXOPERATIONSCPU_CUH

void additionInts(MatrixI M1, MatrixI M2, MatrixI M3);
void additionFloats(MatrixF M1, MatrixF M2, MatrixF M3);
void additionDoubles(MatrixD M1, MatrixD M2, MatrixD M3);

void multiplicationInts(MatrixI M1, MatrixI M2, MatrixI M3);
void multiplicationFloats(MatrixF M1, MatrixF M2, MatrixF M3);
void multiplicationDoubles(MatrixD M1, MatrixD M2, MatrixD M3);

bool compareMatricesInts(MatrixI M1, MatrixI M2);
bool compareMatricesFloats(MatrixF M1, MatrixF M2);
bool compareMatricesDoubles(MatrixD M1, MatrixD M2);

void LUD_Sequential(float *A, int n);
void LUD_Sequential_Partial_Pivoting(float** A, int n);

#endif