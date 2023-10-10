#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../Matrix/Float/matrixFloats.c"
#include "../../Timer/timer.c"

#define MATRIX_SIZE 200

void multiplicationSequential(MatrixFloats M1, MatrixFloats M2, MatrixFloats M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < M1.cols; k++)
            {
                float a = M1.data[i * M1.cols + k];
                float b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }
            M3.data[i * M3.cols + j] = sum;
        }
    }
}

void multiplicationV2(MatrixFloats M1, MatrixFloats M2, MatrixFloats M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        int pos1 = i * M1.cols;
        int pos2 = i * M3.cols;

        for (int j = 0; j < M2.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < M1.cols; k++)
            {
                float a = M1.data[pos1 + k];
                float b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }

            M3.data[pos2 + j] = sum;
        }
    }
}

int main()
{
    // Start measuring time OS spends on process
    Timer timer = createTimer();

    // Initialize matrices
    MatrixFloats M1 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);
    MatrixFloats M2 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);
    MatrixFloats M3 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);

    // Read data into M1 and M2
    populateWithRandomFloats(M1);
    populateWithRandomFloats(M2);

    double executionTimes[100]; // Array to store execution times for 100 iterations

    for (int i = 0; i < 100; i++)
    {
        beginTimer(&timer);                   // Start measuring time for this iteration
        multiplicationSequential(M1, M2, M3); // Perform addition
        executionTimes[i] = endTimer(&timer); // End measuring time for this iteration
    }

    // Open a new file to write result into
    char filename[100];
    snprintf(filename, 100, "Test/Multiplication_Floats_Runtime_Matrix_Size_%d.csv", MATRIX_SIZE);

    FILE *outputFile = fopen(filename, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return 1;
    }
    else
    {
        for (int i = 0; i < 100; i++)
        {
            fprintf(outputFile, "%f\n", executionTimes[i]);
        }
    }

    // Deallocate memory
    free(M1.data);
    free(M2.data);
    free(M3.data);

    // Close result file and end program
    fclose(outputFile);
    return 0;
}
