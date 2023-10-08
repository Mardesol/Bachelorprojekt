#include "..\..\Header Files\matrixInts.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

const int rows = 200;
const int cols = 200;

const int M1Rows = rows;
const int M2Rows = rows;
const int M3Rows = rows;

const int M3Cols = cols;
const int M1Cols = cols;
const int M2Cols = cols;

const size_t memorySize1 = M1Rows * M1Cols * sizeof(int);
const size_t memorySize2 = M2Rows * M2Cols * sizeof(int);
const size_t memorySize3 = M3Rows * M3Cols * sizeof(int);

// Create a matrix on the host
MatrixI createMatrixInts(int rows, int cols) {
    MatrixI matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Allocate host memory for the matrix data
    matrix.data = (int*)malloc(rows * cols * sizeof(int));

    if (matrix.data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    return matrix;
}

// Set all elements in the matrix to hold value 1
void populateWithOnesInts(MatrixI matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = 1;
        }
    }
}

// Generate random integers on the CPU using srand
void populateWithRandomInts(MatrixI matrix) {
    srand(42);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            matrix.data[i * matrix.cols + j] = rand();
        }
    }
}

void printMatrixToFileInts(char* fileName, MatrixI M) {
	FILE* outputFile = fopen(fileName, "w");
	if (outputFile == NULL) {
		perror("Unable to create the output file");
		return;
	}

    // Print the matrix to the file
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            fprintf(outputFile, "%d ", M.data[i * M.cols + j]);
        }
        fprintf(outputFile, "\n");
    }
    
    fclose(outputFile); // Close the file after writing
}

// Comparison for int data type
bool compareMatricesInts(MatrixI M1, MatrixI M2) {
    for (int i = 0; i < M1.rows; i++) {
        for (int j = 0; j < M1.cols; j++) {
            if (M1.data[i * M1.cols + j] != M2.data[i * M1.cols + j]) {
                return false;   // Matrices do not match
            }
        }
    }
    return true;                // Matrices match
}

void initializeMatricesAndMemory(MatrixI& M1, MatrixI& M2, MatrixI& M3) {
    M1 = createMatrixInts(M1Rows, M1Cols);
    M2 = createMatrixInts(M2Rows, M2Cols);
    M3 = createMatrixInts(M3Rows, M3Cols);

    populateWithRandomInts(M1);
    populateWithRandomInts(M2);
}

void allocateMemoryOnGPU(int*& device_M1, int*& device_M2, int*& device_M3) {
    cudaMalloc((void**)&device_M1, memorySize1);
    cudaMalloc((void**)&device_M2, memorySize2);
    cudaMalloc((void**)&device_M3, memorySize3);
}

void copyMatricesToGPU(const MatrixI& M1, const MatrixI& M2, int* device_M1, int* device_M2) {
    cudaMemcpy(device_M1, M1.data, memorySize1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, memorySize2, cudaMemcpyHostToDevice);
}

void freeMemory(int* device_M1, int* device_M2, int* device_M3, MatrixI& M1, MatrixI& M2, MatrixI& M3) {
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    free(M1.data);
    free(M2.data);
    free(M3.data);
}