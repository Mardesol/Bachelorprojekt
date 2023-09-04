#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "../Matrix/matrix.c"
#include "../Timer/timer.c"

void multiplicationSimple(Matrix M1, Matrix M2, Matrix M3)
{
    for(int i = 0; i < M1.rows; i++) {
        for(int j = 0; j < M2.cols; j++) {
            int sum = 0;
            for (int k = 0; k < M1.cols; k++) {
                int a = M1.data[i * M1.cols + k];
                int b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }
            M3.data[i * M3.cols + j] = sum;
        }
    }
}

void multiplicationV2(Matrix M1, Matrix M2, Matrix M3)
{
    for(int i = 0; i < M1.rows; i++) {
            int pos1 = i * M1.cols;
            int pos2 = i * M3.cols;
        
        for(int j = 0; j < M2.cols; j++) {
            int sum = 0;
            
            for (int k = 0; k < M1.cols; k++) {
                int a = M1.data[pos1 + k];
                int b = M2.data[k * M2.cols + j];
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
    beginTimer(&timer);

    Matrix M1;
    Matrix M2;
    Matrix M3;

    M1 = createMatrix(500, 500);
    M2 = createMatrix(500, 500);
    M3 = createMatrix(500, 500);

    // Read data into M1 and M2
    populateWithRandomInts(M1);
    populateWithRandomInts(M2);

    // End measuring time OS spends on process
    endTimer(&timer, "setup", 5);

    // Start measuring time OS spends on process
    beginTimer(&timer);

    // Perform multiplication
    multiplicationSimple(M1, M2, M3);

    // End measuring time OS spends on process
    endTimer(&timer, "multiplication", 8);

    // Start measuring time OS spends on process
    beginTimer(&timer);

    // Open a new file to write result into
    FILE *outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the result file
    for(int i = 0; i < M3.rows; i++) {
        for(int j = 0; j < M3.cols; j++) {
            fprintf(outputFile, "%d ", M3.data[i * M3.cols + j]);
        }
        fprintf(outputFile, "\n");
    }


    // Close result file
    fclose(outputFile);

    free(M1.data);
    free(M2.data);
    free(M3.data);

    // End measuring time OS spends on process
    endTimer(&timer, "shutdown", 8);

    // End program
    return 0;
}
