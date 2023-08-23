#include <stdio.h>
#include <stdlib.h>

int main() 
{
    int N = 3;                // Length of rows and cols
    char file[] = "3.txt";    // Name of file
    
    // Open first matrix file
    FILE *file1 = fopen(file, "r");
    if (file1 == NULL) {
        perror("Unable to open file");
        return 1;
    }
    
    // Open second matrix file
    FILE *file2 = fopen(file, "r");
    if (file2 == NULL) {
        perror("Unable to open file");
        return 1;
    }

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
            fscanf(file1, "%d", &M1[i][j]);
            fscanf(file2, "%d", &M2[i][j]);
        }
    }

    // Close files
    fclose(file1);
    fclose(file2);

    // Perform addition
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            M3[i][j] = M1[i][j] + M2[i][j];
        }
    }

    // Open a new file to write result into
    FILE *outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the output file
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

    // End program
    return 0;
}
