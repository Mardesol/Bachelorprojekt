#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ludMethods.c"
#include "../Matrix/2D/Matrix2D.c"
#include "../Timer/timer.c"

int main(int argc, char *argv[])
{
    int num_sizes = atoi(argv[1]);
    
    int *matrix_sizes = malloc(num_sizes * sizeof(int));
    for (int i = 0; i < num_sizes; i++) {
        matrix_sizes[i] = atoi(argv[i + 2]);
    }

    // Setup
    C_Timer timer = create_C_Timer();
    double **executionTimes;
    executionTimes = (double **) malloc(num_sizes * sizeof(double *));
    for (int i = 0; i < num_sizes; i++) {
        executionTimes[i] = (double *) malloc(100 * sizeof(double));
    }

    // Format file name based on input arguments
    char filename[256];
    sprintf(filename, "Test/LUD_Runtime_");
    for (int i = 0; i < num_sizes; i++) {
        sprintf(filename, "%s_%d", filename, matrix_sizes[i]);
    }
    strcat(filename, ".csv");

    // Create results file in the Test directory
    FILE *outputFile = fopen(filename, "w");
    if (!outputFile)
    {
        perror("Unable to create the output file");
        return 1;
    }

    // Print the header (MATRIX_SIZES[0], MATRIX_SIZES[1], MATRIX_SIZES[2]...)
    for (int s = 0; s < num_sizes; s++)
    {
        fprintf(outputFile, "%d", matrix_sizes[s]);
        if (s != num_sizes - 1)
        {
            fprintf(outputFile, ", ");
        }
    }
    fprintf(outputFile, "\n");

    // Perform all runs
    for (int s = 0; s < num_sizes; s++)
    {
        int size = matrix_sizes[s];

        for (int run = 0; run < 100; run++)
        {
            Matrix2D A = createMatrix2D(size, size);

            populateWithRandomFloats(A);

            beginTimer(&timer);
            LUD_Sequential_Partial_Pivoting(A.data, A.rows);
            double timeTaken = endTimer(&timer);

            executionTimes[s][run] = timeTaken;

            free(A.data);
        }
    }

    // Write execution times to the file
    for (int run = 0; run < 100; run++)
    {
        for (int s = 0; s < num_sizes; s++)
        {
            fprintf(outputFile, "%f", executionTimes[s][run]);
            if (s != num_sizes - 1)
            {
                fprintf(outputFile, ", ");
            }
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);

    for (int i = 0; i < num_sizes; i++) {
        free(executionTimes[i]);
    }
    free(executionTimes);
    free(matrix_sizes);

    return 0;
}