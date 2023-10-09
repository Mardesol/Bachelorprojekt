#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../Matrix/Double/matrixDoubles.c"
#include "../../Timer/timer.c"

#define MATRIX_SIZES {200, 300, 400, 500, 600}
#define NUM_SIZES 5

void multiplicationSequential(MatrixDoubles M1, MatrixDoubles M2, MatrixDoubles M3)
{
    for(int i = 0; i < M1.rows; i++) {
        for(int j = 0; j < M2.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < M1.cols; k++) {
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
    for(int i = 0; i < M1.rows; i++) {
            int pos1 = i * M1.cols;
            int pos2 = i * M3.cols;
        
        for(int j = 0; j < M2.cols; j++) {
            double sum = 0.0;
            
            for (int k = 0; k < M1.cols; k++) {
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
    // Setup
    Timer timer = createTimer();
    const int sizes[NUM_SIZES] = MATRIX_SIZES;
    float executionTimes[NUM_SIZES][100];

    // Create results file in the Test directory
    FILE *outputFile = fopen("Test/Multiplication_Doubles_Runtime_All_Matrices.csv", "w");
    if (!outputFile) {
        perror("Unable to create the output file");
        return 1;
    }

    // Print the header (MATRIX_SIZES[0], MATRIX_SIZES[1], MATRIX_SIZES[2]...)
    for(int s = 0; s < NUM_SIZES; s++) {
        fprintf(outputFile, "%d", sizes[s]);
        if(s != NUM_SIZES - 1) {
            fprintf(outputFile, ", ");
        }
    }
    fprintf(outputFile, "\n");

    // Perform all runs
    for(int s = 0; s < NUM_SIZES; s++) {
        int size = sizes[s];

        for (int run = 0; run < 100; run++) {
            MatrixDoubles M1 = createMatrixDoubles(size, size);
            MatrixDoubles M2 = createMatrixDoubles(size, size);
            MatrixDoubles M3 = createMatrixDoubles(size, size);

            populateWithRandomDoubles(M1);
            populateWithRandomDoubles(M2);

            beginTimer(&timer);
            multiplicationSequential(M1, M2, M3);
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