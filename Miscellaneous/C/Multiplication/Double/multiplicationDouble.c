#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../Matrix/Double/matrixDoubles.c"
#include "../../Timer/timer.c"

#define MATRIX_SIZE 200

void multiplicationSequential(MatrixDoubles M1, MatrixDoubles M2, MatrixDoubles M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < M1.cols; k++)
            {
                double a = M1.data[i * M1.cols + k];
                double b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }
            M3.data[i * M3.cols + j] = sum;
        }
    }
}

void multiplicationV2(MatrixDoubles M1, MatrixDoubles M2, MatrixDoubles M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        int pos1 = i * M1.cols;
        int pos2 = i * M3.cols;

        for (int j = 0; j < M2.cols; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < M1.cols; k++)
            {
                double a = M1.data[pos1 + k];
                double b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }

            M3.data[pos2 + j] = sum;
        }
    }
}

int main()
{
    // Start measuring time OS spends on process
    C_Timer timer = create_C_Timer();

    // Initialize matrices
    MatrixDoubles M1 = createMatrixDoubles(MATRIX_SIZE, MATRIX_SIZE);
    MatrixDoubles M2 = createMatrixDoubles(MATRIX_SIZE, MATRIX_SIZE);
    MatrixDoubles M3 = createMatrixDoubles(MATRIX_SIZE, MATRIX_SIZE);

    // Read data into M1 and M2
    populateWithRandomDoubles(M1);
    populateWithRandomDoubles(M2);

    double executionTimes[100]; // Array to store execution times for 100 iterations

    for (int i = 0; i < 100; i++)
    {
        beginTimer(&timer);                   // Start measuring time for this iteration
        multiplicationSequential(M1, M2, M3); // Perform addition
        executionTimes[i] = endTimer(&timer); // End measuring time for this iteration
    }

    // Open a new file to write result into
    char filename[100];
    snprintf(filename, 100, "Test/Multiplication_Doubles_Runtime_Matrix_Size_%d.csv", MATRIX_SIZE);

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
