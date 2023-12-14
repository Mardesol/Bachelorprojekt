#include <math.h>

void LUD_Sequential(float **A, int n) {
    for (int i = 1; i < n; i++) {
        for (int j = i; j < n; j++) {
            float sum = A[i][j];
            for (int k = 1; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }
        for (int j = i + 1; j < n; j++) {
            float sum = A[j][i];
            for (int k = 1; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
}

void LUD_Sequential_Partial_Pivoting(float** A, int n) {
    for (int i = 1; i < n; i++) {
        int pivotRow = i;
        float maxVal = fabs(A[i][i]);
        for (int p = i + 1; p < n; p++) {
            if (fabs(A[p][i]) > maxVal) {
                maxVal = fabs(A[p][i]);
                pivotRow = p;
            }
        }
        if (pivotRow != i) {
            for (int j = 1; j < n; j++) {
                float temp = A[i][j];
                A[i][j] = A[pivotRow][j];
                A[pivotRow][j] = temp;
            }
        }
        for (int j = i; j < n; j++) {
            float sum = A[i][j];
            for (int k = 1; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }
        for (int j = i + 1; j < n; j++) {
            float sum = A[j][i];
            for (int k = 1; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
}