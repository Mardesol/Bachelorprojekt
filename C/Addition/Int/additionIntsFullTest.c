#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../Matrix/Int/matrixInts.c"
#include "../../Timer/timer.c"

#define MATRIX_SIZES {200, 400, 800, 1600, 3200, 6400}
#define NUM_SIZES 6

void additionSequential(MatrixInts M1, MatrixInts M2, MatrixInts M3)
{
    for(int i = 0; i < M3.rows; i++) {
        for(int j = 0; j < M3.cols; j++) {
            M3.data[i * M3.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M2.cols + j];
        }
    }
}

int main() 
{
    // Setup
    Timer timer = createTimer();
    const int sizes[NUM_SIZES] = MATRIX_SIZES;
    double executionTimes[NUM_SIZES][100];

    // Create results file
    FILE *outputFile = fopen("Test/Addition_Ints_Runtime_All_Matrices.csv", "w");
    if (!outputFile) {
        perror("Unable to create the output file");
        return 1;
    }

    // Print the header
    fprintf(outputFile, "200, 400, 800, 1600, 3200, 6400\n");

    // Perform all runs
    for(int s = 0; s < NUM_SIZES; s++) {
        int size = sizes[s];

        for (int run = 0; run < 100; run++) {
            MatrixInts M1 = createMatrixInts(size, size);
            MatrixInts M2 = createMatrixInts(size, size);
            MatrixInts M3 = createMatrixInts(size, size);

            populateWithRandomFloats(M1);
            populateWithRandomFloats(M2);

            beginTimer(&timer);
            additionSequential(M1, M2, M3);
            double timeTaken = endTimerDouble(&timer);

            executionTimes[s][run] = timeTaken;

            free(M1.data);
            free(M2.data);
            free(M3.data);
        }
    }

    // Write execution times to the file
    for(int run = 0; run < 100; run++) {
        for(int s = 0; s < NUM_SIZES; s++) {
            fprintf(outputFile, "%f", executionTimes[s][run]);
            if(s != NUM_SIZES - 1) {
                fprintf(outputFile, ", ");
            }
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);
    return 0;
}