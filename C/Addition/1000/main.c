#include <stdio.h>
#include <string.h>
#include <stdio.h>   // For file input and output operations
#include <stdlib.h>  // For dynamic memory allocation and deallocation


int main() 
{
    FILE *file1 = fopen("1000.txt", "r");
    if (file1 == NULL) {
        perror("Unable to open file1");
        return 1;
    }
    FILE *file2 = fopen("1000.txt", "r");
    if (file2 == NULL) {
        perror("Unable to open file2");
        return 1;
    }


    int N = 1000;
    int **M1 = (int **)malloc(N * sizeof(int *));
    int **M2 = (int **)malloc(N * sizeof(int *));
    int **M3 = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        M1[i] = (int *)malloc(N * sizeof(int));
        M2[i] = (int *)malloc(N * sizeof(int));
        M3[i] = (int *)malloc(N * sizeof(int));
    }

    // Reading data into M1 and M2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file1, "%d", &M1[i][j]);
            fscanf(file2, "%d", &M2[i][j]);
        }
    }

    fclose(file1);
    fclose(file2);

    // Addition
    for(int i=0; i<1000; i++) {
        for(int j=0; j<1000; j++) {
            M3[i][j] = M1[i][j]+M2[i][j];
            //printf("%d ", M3[i][j]);
        }
        //printf("\n");  
    }

    // Open a new file for writing
    FILE *outputFile = fopen("result.txt", "w");
    
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the output file
    for(int i = 0; i < 1000; i++) {
        for(int j = 0; j < 1000; j++) {
            fprintf(outputFile, "%d ", M3[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);

    // Deallocate memory
    for (int i = 0; i < N; i++) {
        free(M1[i]);
        free(M2[i]);
        free(M3[i]);
    }
    free(M1);
    free(M2);
    free(M3);

    return 0;
}
