#include "..\..\Header Files\matrixDoubles.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

const int rows = 100;
const int cols = 100;

const int M1Rows = rows;
const int M2Rows = rows;
const int M3Rows = rows;

const int M3Cols = cols;
const int M1Cols = cols;
const int M2Cols = cols;

const size_t memorySize1 = M1Rows * M1Cols * sizeof(double);
const size_t memorySize2 = M2Rows * M2Cols * sizeof(double);
const size_t memorySize3 = M3Rows * M3Cols * sizeof(double);

// Create a matrix on the host
MatrixD createMatrixDoubles(int rows, int cols) {
    MatrixD matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (double*)malloc(rows * cols * sizeof(double));

    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1.0
void populateWithOnesDoubles(MatrixD matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1.0;
        }
    }
}

// Generate random floats on the CPU using srand
void populateWithRandomDoubles(MatrixD matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = rand() / (double)rand();
        }
    }
}

void printMatrixToFileDoubles(char* fileName, MatrixD M) {
    FILE* outputFile = fopen(fileName, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return;
    }

    // Print the matrix to the file
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            fprintf(outputFile, "%lf ", M.data[i * M.cols + j]);
        }
        fprintf(outputFile, "\n");
    }
    
    fclose(outputFile); // Close the file after writing
}

// Comparison for double data type
bool compareMatricesDoubles(MatrixD M1, MatrixD M2) {

    //Allowing error margin
    const double ErrorMargin = 1e-5;

    for (int i = 0; i < M1.rows; i++) {
        for (int j = 0; j < M1.cols; j++) {
            if (M1.data[i * M1.cols + j] - M2.data[i * M2.cols + j] > ErrorMargin) {
                return false;   // Matrices do not match
            }
        }
    }
    return true;                // Matrices match
}

// Function to initialize matrices and memory on GPU
void initializeMatricesAndMemory(MatrixD& M1, MatrixD& M2, MatrixD& M3) {
    M1 = createMatrixDoubles(M1Rows, M1Cols);
    M2 = createMatrixDoubles(M2Rows, M2Cols);
    M3 = createMatrixDoubles(M3Rows, M3Cols);

    populateWithRandomDoubles(M1);
    populateWithRandomDoubles(M2);
}

void allocateMemoryOnGPU(double*& device_M1, double*& device_M2, double*& device_M3) {
    cudaMalloc((void**)&device_M1, memorySize1);
    cudaMalloc((void**)&device_M2, memorySize2);
    cudaMalloc((void**)&device_M3, memorySize3);
}

void copyMatricesToGPU(const MatrixD& M1, const MatrixD& M2, double* device_M1, double* device_M2) {
    cudaMemcpy(device_M1, M1.data, memorySize1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, memorySize2, cudaMemcpyHostToDevice);
}

void freeMemory(double* device_M1, double* device_M2, double* device_M3, MatrixD& M1, MatrixD& M2, MatrixD& M3) {
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    free(M1.data);
    free(M2.data);
    free(M3.data);
}