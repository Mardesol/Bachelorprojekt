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
void populateWithRandomFloats(Matrix matrix)
{
    srand(42);

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            //matrix.data[i * matrix.cols + j] = (float)rand() / RAND_MAX;
            matrix.data[i * matrix.cols + j] = (float)rand() / rand();
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
    // const float ErrorMargin = 1e-6f;
    bool matricesMatch = true;

    // Create a matrix for the differences using the createMatrix function
    Matrix Differences = createMatrix(M1.rows, M1.cols);

    for (int i = 0; i < M1.rows; i++) {
        for (int j = 0; j < M1.cols; j++) {
            float diff = fabs(M1.data[i * M1.cols + j] - M2.data[i * M1.cols + j]);
            Differences.data[i * Differences.cols + j] = diff;

            if (diff > ErrorMargin) {
                matricesMatch = false;
            }
        }
    }

    printMatrixToFile(fileName, Differences);

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