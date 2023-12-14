#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ludMethods.c"
#include "../Matrix/2D/Matrix2D.c"
#include "../Timer/timer.c"

#define MATRIX_SIZE 10
#define TINY 1.0e-20 // A small number.

void printLUDMatrixToFile(char* fileName, Matrix2D matrix)
{
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        exit(1);
    }

    // Extracting and printing L
    fprintf(outputFile, "Matrix L:\n");
    for (int i = 1 ; i <= MATRIX_SIZE; i++) {
        for (int j = 1; j <= MATRIX_SIZE; j++) {
            if (i < j) {
                fprintf(outputFile, "0.00 ");
            } else if (i == j) {
                fprintf(outputFile, "1.00 ");
            } else {
                fprintf(outputFile, "%f ", matrix.data[i][j]);
            }
        }
        fprintf(outputFile, "\n");
    }
    
    // Extracting and printing U
    fprintf(outputFile, "\nMatrix U:\n");
    for (int i = 1; i <= MATRIX_SIZE; i++) {
        for (int j = 1; j <= MATRIX_SIZE; j++) {
            if (i > j) {
                fprintf(outputFile, "0.00 ");
            } else {
                fprintf(outputFile, "%f ", matrix.data[i][j]);
            }
        }
        fprintf(outputFile, "\n");
    }
    fclose(outputFile);
}

int main()
{
    // Start measuring time OS spends on process
    C_Timer timer = create_C_Timer();

    // Initialize matrices
    Matrix2D matrix = createMatrix2D(MATRIX_SIZE, MATRIX_SIZE);

    int *indx = (int *)malloc(MATRIX_SIZE * sizeof(int));
    float d;

    // Read data into M1 and M2
    populateWithRandomFloats(matrix);
    //populateWithOnesFloats(matrix);
    printMatrix2DToFile("input.txt", matrix);

    //LUD_Simple(matrix.data, MATRIX_SIZE);
    LUD_Sequential_Partial_Pivoting(matrix.data, MATRIX_SIZE);

    // Write the resulting L and U into a file
    char filename[100];
    snprintf(filename, 100, "Test/result_%d.txt", MATRIX_SIZE);
    printLUDMatrixToFile(filename, matrix);
    printMatrix2DToFile("test.txt", matrix);

    // Deallocate memory
    for (int i = 1; i <= MATRIX_SIZE; i++) {
        free(matrix.data[i]);
    }
    free(indx);

    // End program
    return 0;
}