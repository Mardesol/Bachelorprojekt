#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "additionMethods.c"
#include "../Timer/timer.c"

int main(int argc, char *argv[])
{
    int MRows = atoi(argv[1]);
    int MCols = atoi(argv[2]);
    size_t memorySize = MRows * MCols * sizeof(float);

    // Start measuring time OS spends on process
    C_Timer timer = create_C_Timer();

    // Initialize matrices
    Matrix M1 = create_C_Matrix(MRows, MCols);
    Matrix M2 = create_C_Matrix(MRows, MCols);
    Matrix M3 = create_C_Matrix(MRows, MCols);

    // Read data into M1 and M2
    populateWithRandomFloats(M1);
    populateWithRandomFloats(M2);

    double executionTimes[100];               // Array to store execution times for 100 iterations

    for (int i = 0; i < 100; i++)
    {
        beginTimer(&timer);                   // Start measuring time for this iteration
        additionSequential(M1, M2, M3);       // Perform addition
        executionTimes[i] = endTimer(&timer); // End measuring time for this iteration
    }

    // Open a new file to write result into
    char filename[100];
    snprintf(filename, 100, "Test/Addition_Floats_Runtime_Matrix_Size_%dx%d.csv", MRows, MCols);

    FILE *outputFile = fopen(filename, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return 1;
    }
    else
    {
        for (int i = 0; i < 100; i++)
        {
            fprintf(outputFile, "%f\n", executionTimes[i]);
        }
    }

    // Deallocate memory
    free(M1.data);
    free(M2.data);
    free(M3.data);

    // Close result file and end program
    fclose(outputFile);
    return 0;
}