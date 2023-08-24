#include <stdio.h>

int main() {
    // Define the pattern
    int pattern[] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    
    // Define the size of the matrix
    int rows = 1000;
    int cols = 1000;
    
    // Open a file for writing
    FILE *file = fopen("1000.txt", "w");
    
    if (file == NULL) {
        perror("Unable to create the file");
        return 1;
    }

    // Loop through rows
    for (int i = 0; i < rows; i++) {
        // Loop through columns
        for (int j = 0; j < cols; j++) {
            // Write the pattern element followed by a space
            fprintf(file, "%d ", pattern[j % 9]);
        }
    }

    // Close the file
    fclose(file);

    return 0;
}
