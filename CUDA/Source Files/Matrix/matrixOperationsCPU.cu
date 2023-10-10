#include "..\..\Header Files\matrixOperationsCPU.cuh"

// Addition for int data type
void additionInts(MatrixI M1, MatrixI M2, MatrixI M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M1.cols; j++)
        {
            M3.data[i * M1.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M1.cols + j];
        }
    }
}

// Addition for float data type
void additionFloats(MatrixF M1, MatrixF M2, MatrixF M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M1.cols; j++)
        {
            M3.data[i * M1.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M1.cols + j];
        }
    }
}

// Addition for double data type
void additionDoubles(MatrixD M1, MatrixD M2, MatrixD M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M1.cols; j++)
        {
            M3.data[i * M1.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M1.cols + j];
        }
    }
}

// Multiplication for int data type
void multiplicationInts(MatrixI M1, MatrixI M2, MatrixI M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            M3.data[i * M2.cols + j] = 0.0;
            for (int k = 0; k < M1.cols; k++)
            {
                M3.data[i * M2.cols + j] += M1.data[i * M1.cols + k] * M2.data[k * M2.cols + j];
            }
        }
    }
}

// Multiplication for float data type
void multiplicationFloats(MatrixF M1, MatrixF M2, MatrixF M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            M3.data[i * M2.cols + j] = 0.0;
            for (int k = 0; k < M1.cols; k++)
            {
                M3.data[i * M2.cols + j] += M1.data[i * M1.cols + k] * M2.data[k * M2.cols + j];
            }
        }
    }
}

// Multiplication for double data type
void multiplicationDoubles(MatrixD M1, MatrixD M2, MatrixD M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            M3.data[i * M2.cols + j] = 0.0;
            for (int k = 0; k < M1.cols; k++)
            {
                M3.data[i * M2.cols + j] += M1.data[i * M1.cols + k] * M2.data[k * M2.cols + j];
            }
        }
    }
}