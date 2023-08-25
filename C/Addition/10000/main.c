#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() 
{
    // Start measuring time OS spends on process
    clock_t setupBegin = clock();

    int N = 10000;                // Length of rows and cols
    int pattern[] = {1, 2, 3};

    // Allocate memory for matrices
    int **M1 = (int **)malloc(N * sizeof(int *));
    int **M2 = (int **)malloc(N * sizeof(int *));
    int **M3 = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        M1[i] = (int *)malloc(N * sizeof(int));
        M2[i] = (int *)malloc(N * sizeof(int));
        M3[i] = (int *)malloc(N * sizeof(int));
    }

    // Read data into M1 and M2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M1[i][j] = pattern[j % 9];
            M2[i][j] = pattern[j % 9];
        }
    }

    // End measuring time OS spends on process
    clock_t setupEnd = clock();
    double time_spent1 = (double)(setupEnd - setupBegin) / CLOCKS_PER_SEC;
    printf("Time spent on setup: %f seconds\n", time_spent1);

    // Start measuring time OS spends on process
    clock_t begin = clock();

    // Perform addition
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            M3[i][j] = M1[i][j] + M2[i][j];
        }
    }

    // End measuring time OS spends on process
    clock_t end = clock();
    double time_spent2 = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent on addition: %f seconds\n", time_spent2);

    // Start measuring time OS spends on process
    clock_t ShutdownBegin = clock();

    // Open a new file to write result into
    FILE *outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the result file
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(outputFile, "%d ", M3[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close result file
    fclose(outputFile);

    // Deallocate memory for matrices
    for (int i = 0; i < N; i++) {
        free(M1[i]);
        free(M2[i]);
        free(M3[i]);
    }
    free(M1);
    free(M2);
    free(M3);

    // End measuring time OS spends on process
    clock_t shutdownEnd = clock();
    double time_spent3 = (double)(shutdownEnd - ShutdownBegin) / CLOCKS_PER_SEC;
    printf("Time spent on shutdown: %f seconds\n", time_spent3);

    // End program
    return 0;
}
