#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "../../Matrix/Float/matrixFloats.c"
#include "../../Timer/timer.c"

void additionSequential(MatrixFloats M1, MatrixFloats M2, MatrixFloats M3)
{
    for(int i = 0; i < M3.rows; i++) {
        for(int j = 0; j < M3.cols; j++) {
            M3.data[i * M3.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M2.cols + j];
        }
    }
}

#define MATRIX_SIZE 5000

int main() 
{
    // Start measuring time OS spends on process
    Timer timer = createTimer();
    beginTimer(&timer);

    MatrixFloats M1;
    MatrixFloats M2;
    MatrixFloats M3;

    M1 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);
    M2 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);
    M3 = createMatrixFloats(MATRIX_SIZE, MATRIX_SIZE);

    // Read data into M1 and M2
    populateWithRandomFloats(M1);
    populateWithRandomFloats(M2);

    // Array to store execution times for 100 iterations
    float executionTimes[100];

    // End measuring time OS spends on process
    endTimer(&timer, "setup", 5);

    for (int i = 0; i < 100; i++) {
        // Start measuring time for this iteration
        beginTimer(&timer);
        // Perform addition
        additionSequential(M1, M2, M3);
        // End measuring time for this iteration
        executionTimes[i] = endTimerFloat(&timer);
    }

    // Open a new file to write result into
    
    char filename[100];
    snprintf(filename, 100, "FloatsResult%d.txt", MATRIX_SIZE);

    FILE *outputFile = fopen(filename, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    } else {
        for(int i = 0; i < 100; i++) {
            fprintf(outputFile, "%f\n", executionTimes[i]);
        }
    }
    // Close result file
    fclose(outputFile);


    // Start measuring time OS spends on process
    beginTimer(&timer);

    free(M1.data);
    free(M2.data);
    free(M3.data);

    // End measuring time OS spends on process
    endTimer(&timer, "shutdown", 8);

    // End program
    return 0;
}
