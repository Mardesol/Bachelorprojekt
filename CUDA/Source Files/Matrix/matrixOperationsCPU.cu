#include "..\..\Header Files\matrixOperationsCPU.cuh"

// Addition for int data type
void additionInt(int* M1, int* M2, int* M3, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            M3[i * MCols + j] = M1[i * MCols + j] + M2[i * MCols + j];
        }
    }
}

// Addition for float data type
void additionFloat(float* M1, float* M2, float* M3, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            M3[i * MCols + j] = M1[i * MCols + j] + M2[i * MCols + j];
        }
    }
}

// Addition for double data type
void additionDouble(double* M1, double* M2, double* M3, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            M3[i * MCols + j] = M1[i * MCols + j] + M2[i * MCols + j];
        }
    }
}

// Multiplication for int data type
void multiplicationInt(int* M1, int* M2, int* M3, int M1Rows, int M1Cols, int M2Cols) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M2Cols; j++) {
            M3[i * M2Cols + j] = 0;
            for (int k = 0; k < M1Cols; k++) {
                M3[i * M2Cols + j] += M1[i * M1Cols + k] * M2[k * M2Cols + j];
            }
        }
    }
}

// Multiplication for float data type
void multiplicationFloat(float* M1, float* M2, float* M3, int M1Rows, int M1Cols, int M2Cols) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M2Cols; j++) {
            M3[i * M2Cols + j] = 0.0f;
            for (int k = 0; k < M1Cols; k++) {
                M3[i * M2Cols + j] += M1[i * M1Cols + k] * M2[k * M2Cols + j];
            }
        }
    }
}

// Multiplication for double data type
void multiplicationDouble(double* M1, double* M2, double* M3, int M1Rows, int M1Cols, int M2Cols) {
    for (int i = 0; i < M1Rows; i++) {
        for (int j = 0; j < M2Cols; j++) {
            M3[i * M2Cols + j] = 0.0;
            for (int k = 0; k < M1Cols; k++) {
                M3[i * M2Cols + j] += M1[i * M1Cols + k] * M2[k * M2Cols + j];
            }
        }
    }
}

// Comparison for int data type
bool compareMatricesInt(int* M1, int* M2, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            if (M1[i * MCols + j] != M2[i * MCols + j]) {
                return false;  // Matrices do not match
            }
        }
    }
    return true;  // Matrices match
}

// Comparison for float data type
bool compareMatricesFloat(float* M1, float* M2, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            if (M1[i * MCols + j] != M2[i * MCols + j]) {
                return false;  // Matrices do not match
            }
        }
    }
    return true;  // Matrices match
}

// Comparison for double data type
bool compareMatricesDouble(double* M1, double* M2, int MRows, int MCols) {
    for (int i = 0; i < MRows; i++) {
        for (int j = 0; j < MCols; j++) {
            if (M1[i * MCols + j] != M2[i * MCols + j]) {
                return false;  // Matrices do not match
            }
        }
    }
    return true;  // Matrices match
}