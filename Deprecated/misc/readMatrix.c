#include <stdio.h>
#include <stdlib.h>
#include "readMatrix.h"

// Function to read values from file into a matrix
int readMatrixFromFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening the file");
        exit(1);
    }

    printf("Starting.. \n \n");

    int rows, cols;

    FILE *file1 = fopen(filename, "r");
    fscanf(file1, "%d %d", &rows, &cols);

    int matrix[rows][cols];

    // Read values into the matrix
    char ignore[4];
    fgets(ignore, 4, file);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }

    fclose(file);

    printMatrix(rows, cols, matrix);

    return 0;
}

// Function to print a matrix
void printMatrix(int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(char *path) {
    int rows, cols;

    readMatrixFromFile(path, rows, cols, matrix);
    
    printf("Matrix 1: \n");
    

    return 0;
}
