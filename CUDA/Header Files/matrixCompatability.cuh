#ifndef MATRIXCOMPATABILITY_CUH
#define MATRIXCOMPATABILITY_CUH

bool isCompatibleForAddition(int M1Rows, int M1Cols, int M2Rows, int M2Cols);
bool isCompatibleForMultiplication(int M1Cols, int M2Rows);
#endif