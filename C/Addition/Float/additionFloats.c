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

    M1 = createMatrixFloats(500, 500);
    M2 = createMatrixFloats(500, 500);
    M3 = createMatrixFloats(500, 500);

    // Read data into M1 and M2
    populateWithOnesFloats(M1);
    populateWithOnesFloats(M2);
    //populateWithRandomFloats(M1);
    //populateWithRandomFloats(M2);

    // End measuring time OS spends on process
    endTimer(&timer, "setup", 5);

    // Start measuring time OS spends on process
    beginTimer(&timer);

    // Perform addition
    additionSimple(M1, M2, M3);

    // End measuring time OS spends on process
    endTimer(&timer, "addition", 8);

    // Start measuring time OS spends on process
    beginTimer(&timer);

    // Open a new file to write result into
    FILE *outputFile = fopen("FloatsResult.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the result file
    for(int i = 0; i < M3.rows; i++) {
        for(int j = 0; j < M3.cols; j++) {
            fprintf(outputFile, "%f ", M3.data[i * M3.cols + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close result file
    fclose(outputFile);

    // Deallocate memory for matrices
    // for (int i = 0; i < N; i++) {
    //     free(M1[i]);
    //     free(M2[i]);
    //     free(M3[i]);
    // }
    free(M1.data);
    free(M2.data);
    free(M3.data);

    // End measuring time OS spends on process
    endTimer(&timer, "shutdown", 8);

    // End program
    return 0;
}
