#include <stdio.h>
#include <string.h>
#include "readMatrix.h"
#include "Operations/addition.h"



// argc is the number of arguments the method will be provided
// argv are the arguments the user provides, like 'addition 300.txt'
int main(int argc, char *argv[]) 
{
    
    // Get the operation argument
    char *operation = argv[1];
    
    // Read the matrix from file
    int M1 = readMatrixFromFile(argv[2]);
    int M2 = readMatrixFromFile(argv[3]);
    

    if (strcmp(operation, "add") == 0) {
        int result = add(3, 3, M1, M2);

    } else if (strcmp(operation, "mul") == 0) {
        //mul();
        return 1;

    } else if (strcmp(operation, "lud") == 0) {
        //lud();
        return 1;

    } else if (strcmp(operation, "svd") == 0) {
        //svd();
        return 1;

    } else {
        return 1;
    }

    return 0;
}