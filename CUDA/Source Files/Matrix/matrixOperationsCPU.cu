#include "..\..\Header Files\matrixOperationsCPU.cuh"

// Addition for float data type
void addition(Matrix M1, Matrix M2, Matrix M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M1.cols; j++)
        {
            M3.data[i * M1.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M1.cols + j];
        }
    }
}

// Multiplication for float data type
void multiplication(Matrix M1, Matrix M2, Matrix M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            M3.data[i * M2.cols + j] = 0.0;
            for (int k = 0; k < M1.cols; k++)
            {
                M3.data[i * M2.cols + j] += M1.data[i * M1.cols + k] * M2.data[k * M2.cols + j];
            }
        }
    }
}

void LUD_Sequential(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }
        for (int j = i + 1; j < n; j++) {
            float sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}

void LUD_Sequential_Partial_Pivoting(float* A, int n) {
    for (int i = 0; i < n; i++) {
        int pivotRow = i;
        float maxVal = fabs(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabs(A[p * n + i]) > maxVal) {
                maxVal = fabs(A[p * n + i]);
                pivotRow = p;
            }
        }

        if (pivotRow != i) {
            for (int j = 0; j < n; j++) {
                float temp = A[i * n + j];
                A[i * n + j] = A[pivotRow * n + j];
                A[pivotRow * n + j] = temp;
            }
        }

        for (int j = i; j < n; j++) {
            float sum = A[i * n + j];
            for (int k = 0; k < i; k++) {
                sum -= A[i * n + k] * A[k * n + j];
            }
            A[i * n + j] = sum;
        }

        for (int j = i + 1; j < n; j++) {
            float sum = A[j * n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[j * n + k] * A[k * n + i];
            }
            A[j * n + i] = sum / A[i * n + i];
        }
    }
}



