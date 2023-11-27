#include "..\..\Header Files\matrix.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

// Create a matrix on the host
Matrix createMatrix(int rows, int cols)
{
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (float *)malloc(rows * cols * sizeof(float));

    if (matrix.data == NULL)
    {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1.0f
void populateWithOnes(Matrix matrix)
{
    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            matrix.data[i * matrix.cols + j] = 1.0f; // Change to float
        }
    }
}

// Generate random floats on the CPU using srand
void populateWithRandomFloats1(Matrix matrix)
{
    srand(42);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            float rand1 = rand();
            float rand2 = rand()+0.00001f;
            float f = (float)rand1 / rand2;
            if (isnan(f)) {
                printf("I have created a nan: %f when %f / %f\n", f, rand1, rand2);
            }
            else if (isinf(f)) {
                printf("I have created a inf: %f when %f / %f\n", f, rand1, rand2);
            }
            //matrix.data[i * matrix.cols + j] = (float)rand() / RAND_MAX;
            //matrix.data[i * matrix.cols + j] = (float)rand() / rand();
            matrix.data[i * matrix.cols + j] = f;
        }
    }
}

void populateWithRandomFloats(Matrix matrix)
{
    srand(42);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            //float rand1 = rand();
            float rand2 = rand() + 0.00001f;
            float f = rand2;
            matrix.data[i * matrix.cols + j] = f;
        }
    }
}

void printMatrixToFile(char *fileName, Matrix M)
{
    FILE *outputFile = fopen(fileName, "w");
    if (outputFile == NULL)
    {
        perror("Unable to create the output file");
        return;
    }

    // Print the matrix to the file
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            fprintf(outputFile, "%f ", M.data[i * M.cols + j]);
        }
        fprintf(outputFile, "\n");
    }

    fclose(outputFile); // Close the file after writing
}

bool compareMatrices(Matrix M1, Matrix M2)
{
    const float ErrorMargin = (float)1;
    //const float ErrorMargin = 1e-6f;

    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M1.cols; j++)
        {
            if (fabs(M1.data[i * M1.cols + j] - M2.data[i * M1.cols + j]) > ErrorMargin)
            {
                return false; // Matrices do not match
            }
        }
    }
    return true; // Matrices match
}

bool compareAndPrintDifferences(Matrix M1, Matrix M2, char* fileName) {
    const float ErrorMargin = (float)1;
    float sum = 0.0f;
    int bads = 0;
    // const float ErrorMargin = 1e-6f;
    bool matricesMatch = true;

    // Create a matrix for the differences using the createMatrix function
    Matrix Differences = createMatrix(M1.rows, M1.cols);

    for (int i = 0; i < M1.rows; i++) {
        for (int j = 0; j < M1.cols; j++) {
            float diff = fabs(M1.data[i * M1.cols + j] - M2.data[i * M1.cols + j]);
            /*if (isnan(diff)) {
                printf("nan detected at index %d, %d, when calculating %f - %f\n", i, j, M1.data[i * M1.cols + j], M2.data[i * M1.cols + j]);
            }*/
            sum += diff;
            Differences.data[i * Differences.cols + j] = diff;

            if (diff > ErrorMargin) {
                bads += 1;
                matricesMatch = false;
            }
        }
    }

    printMatrixToFile(fileName, Differences);

    FILE* outputFile = fopen(fileName, "a");
    fprintf(outputFile, "\n");
    fprintf(outputFile, "Sum diff: %f ", sum);
    fprintf(outputFile, "\n");
    fprintf(outputFile, "Bad results: %d", bads);
    fprintf(outputFile, "\n");
    fprintf(outputFile, "Percentage of results being bad: %f", ((float)(bads)/(M1.rows*M1.cols))*100.0f);

    free(Differences.data);

    return matricesMatch;
}


Matrix twoDim_to_MatrixF(float** twoDim, int rows, int cols)
{
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (float *)malloc(rows * cols * sizeof(float));

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix.data[i * cols + j] = twoDim[i][j];
        }
    }

    return matrix;
}

float** MatrixF_to_twoDim(Matrix matrix) {
    float **twoDim = (float **)malloc(matrix.rows * sizeof(float *));
    for(int i = 0; i < matrix.rows; ++i) {
        twoDim[i] = (float *)malloc(matrix.cols * sizeof(float));
    }

    for(int i = 0; i < matrix.rows; ++i) {
        for(int j = 0; j < matrix.cols; ++j) {
            twoDim[i][j] = matrix.data[i * matrix.cols + j];
        }
    }
    return twoDim;
}

void pivotMatrix(float* A, int n) {
    for (int i = 0; i < n; i++) {

        // Find pivot row
        int pivotRow = i;
        float maxVal = fabsf(A[i * n + i]);

        for (int p = i + 1; p < n; p++) {
            if (fabsf(A[p * n + i]) > maxVal) {
                maxVal = fabsf(A[p * n + i]);
                pivotRow = p;
            }
        }

        // Swap rows if needed
        if (pivotRow != i) {
            for (int j = 0; j < n; j++) {
                float temp = A[i * n + j];
                A[i * n + j] = A[pivotRow * n + j];
                A[pivotRow * n + j] = temp;
            }
        }
    }
}

void initializeMatricesAndMemory(Matrix &M1, Matrix &M2, Matrix &M3, int M1Rows, int M1Cols, int M2Rows, int M2Cols, int M3Rows, int M3Cols)
{
    M1 = createMatrix(M1Rows, M1Cols);
    M2 = createMatrix(M2Rows, M2Cols);
    M3 = createMatrix(M2Rows, M2Cols);

    populateWithRandomFloats(M1);
    populateWithRandomFloats(M2);
}

void allocateMemoryOnGPU(float *&device_M1, float *&device_M2, float *&device_M3, size_t memorySize1, size_t memorySize2, size_t memorySize3)
{
    cudaMalloc((void **)&device_M1, memorySize1);
    cudaMalloc((void **)&device_M2, memorySize2);
    cudaMalloc((void **)&device_M3, memorySize3);
}

void copyMatricesToGPU(const Matrix &M1, const Matrix &M2, float *device_M1, float *device_M2, size_t memorySize1, size_t memorySize2)
{
    cudaMemcpy(device_M1, M1.data, memorySize1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, memorySize2, cudaMemcpyHostToDevice);
}

void freeMemory(float *device_M1, float *device_M2, float *device_M3, Matrix &M1, Matrix &M2, Matrix &M3)
{
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    free(M1.data);
    free(M2.data);
    free(M3.data);
}