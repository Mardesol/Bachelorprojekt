#include "multiplicationIntsKernels.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Function to measure kernel execution time
float measureKernelExecutionTime(
	void (*kernel)(int *, int *, int *, int, int, int),
	int *M1, int *M2, int *M3, int M1Rows, int M1Cols, int M2Cols,
	dim3 gridDim, dim3 blockDim)
{
	Timer timer = createTimer();
	beginTimer(timer);

	cudaDeviceSynchronize();
	kernel<<<gridDim, blockDim>>>(M1, M2, M3, M1Rows, M1Cols, M2Cols);
	cudaDeviceSynchronize();

	return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
	float *executionTimes,
	void (*kernel)(int *, int *, int *, int, int, int),
	int *M1, int *M2, int *M3, int M1Rows, int M1Cols, int M2Cols,
	dim3 gridDim, dim3 blockDim)
{
	for (int i = 0; i < 100; i++)
	{
		// Measure execution time for the kernel
		float time = measureKernelExecutionTime(kernel, M1, M2, M3, M1Rows, M1Cols, M2Cols, gridDim, blockDim);
		executionTimes[i] = time;
	}
}

int main(int argc, char* argv[])
{
	int M1Rows = atoi(argv[1]);
    int M1Cols = atoi(argv[2]);
	int M2Rows = atoi(argv[3]);
    int M2Cols = atoi(argv[4]);
	int M3Rows = M1Rows;
    int M3Cols = M2Cols;

    size_t memorySize1 = M1Rows * M1Cols * sizeof(int);
	size_t memorySize2 = M2Rows * M2Cols * sizeof(int);
	size_t memorySize3 = M3Rows * M3Cols * sizeof(int);

	if (!isCompatibleForMultiplication(M1Cols, M2Rows))
	{
		perror("Matrices must be compatible");
		return 1;
	}

	// Timer measure time spent on a process
	Timer timer = createTimer();

	beginTimer(timer);
	MatrixI M1, M2, M3;
	int *device_M1, *device_M2, *device_M3;
	initializeMatricesAndMemory(M1, M2, M3, M1Rows, M1Cols, M2Rows, M2Cols, M3Rows, M3Cols);
	allocateMemoryOnGPU(device_M1, device_M2, device_M3, memorySize1, memorySize2, memorySize3);
	copyMatricesToGPU(M1, M2, device_M1, device_M2, memorySize1, memorySize2);
	endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(16, 16);

	if (M3Rows <= 16 && M3Cols <= 16)
	{
		blockDim = dim3(M3Cols, M3Rows); // Use matrix size for smaller matrices
	}

	dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

	// Create an array to store execution times for each kernel
	float executionTimes[3][100]; // 3 kernels, 100 executions each

	// Measure and record execution times
	measureExecutionTimes(executionTimes[0], Sequential, 			device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols, gridDim, blockDim);
	measureExecutionTimes(executionTimes[1], Parallel,				device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols, gridDim, blockDim);
	measureExecutionTimes(executionTimes[2], SharedMemoryAndTiling, device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Cols, gridDim, blockDim);

	// Copy the result matrix from device to host
	cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

	// Open a new file to write the result into
	char fileName[100];																			  // Max length filename (Just needs to be long enough)
	sprintf(fileName, "Test/Multiplication_Ints_Execution_Times_Size_%dx%d.csv", M3Rows, M3Cols); // Customize filename to reflect size of result matrix
	FILE *outputFile = fopen(fileName, "w");
	if (outputFile == NULL)
	{
		perror("Unable to create the output file");
		return 1;
	}

	// Write execution times to the output file in separate columns
	fprintf(outputFile, "Sequential,Parallel,SharedMemoryAndTilling\n");
	for (int i = 0; i < 100; i++)
	{
		fprintf(outputFile, "%f,%f,%f\n",
				executionTimes[0][i],
				executionTimes[1][i],
				executionTimes[2][i]);
	}

	// Close the output file
	fclose(outputFile);

	freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

	// Exit program
	return 0;
}