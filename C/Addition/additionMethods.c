#include "../Matrix/matrix.c"

void additionSequential(Matrix M1, Matrix M2, Matrix M3)
{
    for (int i = 0; i < M3.rows; i++)
    {
        for (int j = 0; j < M3.cols; j++)
        {
            M3.data[i * M3.cols + j] = M1.data[i * M1.cols + j] + M2.data[i * M2.cols + j];
        }
    }
}