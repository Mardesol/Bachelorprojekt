#ifndef MATRIXOPERATIONSCPU_CUH
#define MATRIXOPERATIONSCPU_CUH

void additionInt(int* M1, int* M2, int* M3, int MRows, int MCols);
void additionFloat(float* M1, float* M2, float* M3, int MRows, int MCols);
void additionDouble(double* M1, double* M2, double* M3, int MRows, int MCols);

void multiplicationInt(int* M1, int* M2, int* M3, int M1Rows, int M1Cols, int M2Cols);
void multiplicationFloat(float* M1, float* M2, float* M3, int M1Rows, int M1Cols, int M2Cols);
void multiplicationDouble(double* M1, double* M2, double* M3, int M1Rows, int M1Cols, int M2Cols);

#endif