#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "../../Matrix/Float/matrixFloats.c"
#include "../../Timer/timer.c"

void additionSimple(MatrixFloats M1, MatrixFloats M2, MatrixFloats M3)
{
    for(int i = 0; i < M3.rows; i++) {
        for(int j = 0; j < M3.cols; j++) {
            M3.data[i * M3.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M2.cols + j];
        }
    }
}

int main() 
{
    // Start measuring time OS spends on process
    Timer timer = createTimer();
    beginTimer(&timer);

    MatrixFloats M1;
    MatrixFloats M2;
    MatrixFloats M3;

    M1 = createMatrixFloats(2000, 2000);
    M2 = createMatrixFloats(2000, 2000);
    M3 = createMatrixFloats(2000, 2000);

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
        additionSimple(M1, M2, M3);
        // End measuring time for this iteration
        executionTimes[i] = endTimerFloat(&timer);
    }

    // Open a new file to write result into
    FILE *outputFile = fopen("FloatsResult.txt", "w");
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
