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
                float sum = M1.data[i * M1.cols + k] * M2.data[k * M2.cols + j];
                /*if (isnan(sum)) {
                    printf("nan detected on index %d, %d, when calculating %f * %f, with indices %d and %d\n", i, j, M1.data[i * M1.cols + k], M2.data[k * M2.cols + j], i * M1.cols + k, k * M2.cols + j);
                }*/
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
    int* pivotIndices = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        int pivotRow = i;
        float maxVal = fabs(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabs(A[p * n + i]) > maxVal) {
                maxVal = fabs(A[p * n + i]);
                pivotRow = p;
            }
        }
        pivotIndices[i] = pivotRow;

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

    printf("Host Pivot Indices:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", pivotIndices[i]);
    }
    printf("\n");
}

void applyPivoting(float* matrix, int* pivotIndices, int n) {
    float* tempRow = (float*)malloc(n * sizeof(float));

    for (int i = n - 1; i >= 0; --i) {
        //printf("index %d is %d \n", i, pivotIndices[i]);
        if (pivotIndices[i] != i) {
            //printf("Im swapping row %d with row %d \n", i, pivotIndices[i]);
            // Swap the entire current row with the pivot row
            for (int col = 0; col < n; ++col) {
                tempRow[col] = matrix[i * n + col];
                matrix[i * n + col] = matrix[pivotIndices[i] * n + col];
                matrix[pivotIndices[i] * n + col] = tempRow[col];
            }
        }
    }

    free(tempRow);
}

void separateLU(float* combinedLU, float* L, float* U, int n) {
    // Initialize L and U matrices
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                U[i * n + j] = combinedLU[i * n + j]; // Diagonal element belongs to U
                L[i * n + j] = 1.0f; // Initialize L's diagonal with 1's
            }
            else if (i > j) {
                L[i * n + j] = combinedLU[i * n + j]; // Lower part of L
                U[i * n + j] = 0.0f; // Initialize U with zeros in the lower part
            }
            else {
                L[i * n + j] = 0.0f; // Initialize L with zeros in the upper part
                U[i * n + j] = combinedLU[i * n + j]; // Upper part of U
            }
        }
    }
}

