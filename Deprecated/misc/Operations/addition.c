#include <stdio.h>
#include "addition.h"

int add(int rows, int cols, int M1[rows][cols], int M2[rows][cols]) {
    int M3[rows][cols];

    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            M3[i][j] = M1[i][j]+M2[i][j];
            printf("%d ", M3[i][j]);
        }
        printf("\n");  
    }

    return 0;
}
