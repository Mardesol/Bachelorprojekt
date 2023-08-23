#include <stdio.h>
#include <string.h>

int main() 
{
    FILE *file1 = fopen("3.txt", "r");
    
    int M1[3][3]; 
    int M2[3][3];
    int M3[3][3];

    // Read data into M1
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fscanf(file1, "%d", &M1[i][j]);
        }
    }

    fclose(file1);

    FILE *file2 = fopen("3.txt", "r");

    // Read data into M2
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fscanf(file2, "%d", &M2[i][j]);
        }
    }

    fclose(file2);

    // Addition
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            M3[i][j] = M1[i][j]+M2[i][j];
        }
    }

    // Open a new file for writing
    FILE *outputFile = fopen("result.txt", "w");
    
    
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write M3 to the output file
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            fprintf(outputFile, "%d ", M3[i][j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);

    return 0;
}
