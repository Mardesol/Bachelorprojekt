#include "Matrix2DFloats.h"
#include <stdio.h>
#include <stdlib.h>

// Create a matrix
struct Matrix2D createMatrix2D(int rows, int cols)
{
    struct Matrix2D matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    matrix.data = (float **)malloc((rows + 1) * sizeof(float*));  // Adjusted here
    if (matrix.data == NULL)
    {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    for (int i = 1; i <= rows; i++)  // Adjusted loop
    {
        matrix.data[i] = (float *)malloc((cols + 1) * sizeof(float));  // Adjusted here
        if (matrix.data[i] == NULL)
        {
            printf("Memory allocation failed for row %d.\n", i);
            exit(1);
        }
    }

    return matrix;
}

// Set all indices in matrix to hold value 1
void populateWithOnes(Matrix2D matrix)
{
    for (int i = 1; i <= matrix.rows; i++) 
    {
        for (int j = 1; j <= matrix.cols; j++)  
        {
            matrix.data[i][j] = 1;
        }
    }
}

void populateWithRandomFloats(Matrix2D matrix)
{
    srand(42);

    for (int i = 1; i <= matrix.rows; i++)  
    {
        for (int j = 1; j <= matrix.cols; j++) 
        {
            matrix.data[i][j] = 100.0f + (900.0f * rand() / (float)RAND_MAX);
        }
    }
}

void printMatrix2DToFile(char *fileName, Matrix2D matrix)
{
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return;
    }

    for (int i = 1; i <= matrix.rows; i++)  // Adjusted loop
    {
        for (int j = 1; j <= matrix.cols; j++)  // Adjusted loop
        {
            fprintf(outputFile, "%f ", matrix.data[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);
}