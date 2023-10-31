void LUD_Sequential(float **A, int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = i; j <= n; j++) {
            
            float sum = A[i][j];
            for (int k = 1; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        for (int j = i+1; j <= n; j++) {
            
            float sum = A[j][i];
            for (int k = 1; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
}

void LUD_Sequential_Partial_Pivoting(float** A, int n) {
    for (int i = 1; i <= n; i++) {

        //Find pivot row                                    // Pivot row = row with the highest absolute value on the diagonal of the unworked rows
        int pivotRow = i;                                   //Set pivot row to current row
        float maxVal = fabs(A[i][i]);                       //Set max value to current rows diagonal
        for (int p = i + 1; p <= n; p++) {                  //Check if another row below has a higher absolut value on the diagonal
            if (fabs(A[p][i]) > maxVal) {
                maxVal = fabs(A[p][i]);                     //If yes, set that element to new max
                pivotRow = p;                               //And that row to the pivot row
            }
        }

        //Swap rows if needed
        if (pivotRow != i) {                                //Checks if current row is not the pivot row
            for (int j = 1; j <= n; j++) {                  //If not, swap the current row with the pivot row
                float temp = A[i][j];                       
                A[i][j] = A[pivotRow][j];
                A[pivotRow][j] = temp;
            }
        }

        //Perform LUD
        for (int j = i; j <= n; j++) {

            float sum = A[i][j];
            for (int k = 1; k < i; k++) {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        for (int j = i + 1; j <= n; j++) {

            float sum = A[j][i];
            for (int k = 1; k < i; k++) {
                sum -= A[j][k] * A[k][i];
            }
            A[j][i] = sum / A[i][i];
        }
    }
}

/**
 * Given a matrix a[1..n][1..n], this routine replaces it by the LU decomposition of a rowwise
 * permutation of itself. a and n are input. a is output, arranged as in equation (2.3.14) above;
 * indx[1..n] is an output vector that records the row permutation effected by the partial
 * pivoting; d is output as ±1 depending on whether the number of row interchanges was even
 * or odd, respectively. This routine is used in combination with lubksb to solve linear equations
 * or invert a matrix.
 */
 /**
void LUD_Pivoting_Crout(float **a, int n, int *indx, float *d) {
    int i, imax, j, k;
    float big, dum, sum, temp;
    float *vv;                  // vv stores the implicit scaling of each row

    vv = vector(1, n);
    *d = 1.0;                   // No row interchanges yet.

    // Loop over rows to get the implicit scaling information
    for (i = 1; i <= n; i++) {
        big = 0.0;
        for (j = 1; j <= n; j++) {
            if ((temp = fabs(a[i][j])) > big) big = temp;
        }
        if (big == 0.0) nrerror("Singular matrix in routine ludcmp");   // No nonzero largest element
        vv[i] = 1.0 / big;                                              // Save the scaling
    }

    // Loop over columns
    for (j = 1; j <= n; j++) {
        for (i = 1; i < j; i++) {
            sum = a[i][j];
            for (k = 1; k < i; k++) sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
        }
        big = 0.0;

        // Search for largest pivot element
        for (i = j; i <= n; i++) {
            sum = a[i][j];
            for (k = 1; k < j; k++) sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            if ((dum = vv[i] * fabs(sum)) >= big) {            // Is the figure of merit for the pivot better than the best so far?
                big = dum;
                imax = i;
            }
        }
        if (j != imax) {                    // Do we need to interchange rows?
            for (k = 1; k <= n; k++) {      // Yes, do so...
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            *d = -(*d);                     // ...and change the parity of d.
            vv[imax] = vv[j];               // Also interchange the scale factor.
        }

        indx[j] = imax;
        
        if (a[j][j] == 0.0) a[j][j] = TINY; // A small number

        // Divide by pivot element
        if (j != n) {
            dum = 1.0 / (a[j][j]);
            for (i = j + 1; i <= n; i++) a[i][j] *= dum;
        }
    }
    
    free_vector(vv, 1, n);
}
*/