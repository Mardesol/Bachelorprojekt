// V3

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "../../Matrix/Matrix/matrix.h"
#include "../../Matrix/Matrix/matrix.c"

int main() 
{
    // Start measuring time OS spends on process
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    Matrix M1;
    Matrix M2;
    Matrix M3;

    M1 = createMatrix(400, 600);
    M2 = createMatrix(700, 500);
    M3 = createMatrix(400, 500);

    populateWithOnes(&M1);
    populateWithOnes(&M2);

    // End measuring time OS spends on process
    gettimeofday(&end, 0);
    long seconds1 = end.tv_sec - begin.tv_sec;
    long microseconds1 = end.tv_usec - begin.tv_usec;
    double elapsed1 = seconds1 + microseconds1 * 1e-6;
    printf("Time spent on setup: %f seconds\n", elapsed1);

    // Start measuring time OS spends on process
    gettimeofday(&begin, 0);

    // Perform multiplication
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

    // End measuring time OS spends on process
    gettimeofday(&end, 0);
    long seconds2 = end.tv_sec - begin.tv_sec;
    long microseconds2 = end.tv_usec - begin.tv_usec;
    double elapsed2 = seconds2 + microseconds2 * 1e-6;
    printf("Time spent on multiplication: %f seconds\n", elapsed2);

    // Start measuring time OS spends on process
    gettimeofday(&begin, 0);

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

    // Deallocate memory for matrices
    free(M1.data);
    free(M2.data);
    free(M3.data);

    // End measuring time OS spends on process
    gettimeofday(&end, 0);
    long seconds3 = end.tv_sec - begin.tv_sec;
    long microseconds3 = end.tv_usec - begin.tv_usec;
    double elapsed3 = seconds3 + microseconds3 * 1e-6;
    printf("Time spent on shutdown: %f seconds\n", elapsed3);

    // End program
    return 0;
}
