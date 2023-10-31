#include "../Matrix/matrix.c"

void multiplicationSequential(Matrix M1, Matrix M2, Matrix M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        for (int j = 0; j < M2.cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < M1.cols; k++)
            {
                float a = M1.data[i * M1.cols + k];
                float b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }
            M3.data[i * M3.cols + j] = sum;
        }
    }
}

void multiplicationV2(Matrix M1, Matrix M2, Matrix M3)
{
    for (int i = 0; i < M1.rows; i++)
    {
        int pos1 = i * M1.cols;
        int pos2 = i * M3.cols;

        for (int j = 0; j < M2.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < M1.cols; k++)
            {
                float a = M1.data[pos1 + k];
                float b = M2.data[k * M2.cols + j];
                sum = sum + (a * b);
            }

            M3.data[pos2 + j] = sum;
        }
    }
}