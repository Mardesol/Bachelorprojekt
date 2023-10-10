#include "multiplicationFloatsKernels.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

// Function to measure kernel execution time
float measureKernelExecutionTime(
	void (*kernel)(float *, float *, float *),
	float *M1, float *M2, float *M3,
	dim3 gridDim, dim3 blockDim)
{
	Timer timer = createTimer();
	beginTimer(timer);

	cudaDeviceSynchronize();
	kernel<<<gridDim, blockDim>>>(M1, M2, M3);
	cudaDeviceSynchronize();

	return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
	float *executionTimes,
	void (*kernel)(float *, float *, float *),
	float *M1, float *M2, float *M3,
	dim3 gridDim, dim3 blockDim)
{
	for (int i = 0; i < 100; i++)
	{
		// Measure execution time for the kernel
		float time = measureKernelExecutionTime(kernel, M1, M2, M3, gridDim, blockDim);
		executionTimes[i] = time;
	}
}

int main()
{
	if (!isCompatibleForMultiplication(M1Cols, M2Rows))
	{
		perror("Matrices must be compatible");
		return 1;
	}

	// Timer measure time spent on a process
	Timer timer = createTimer();

	beginTimer(timer);
	MatrixF M1, M2, M3;
	float *device_M1, *device_M2, *device_M3;
	initializeMatricesAndMemory(M1, M2, M3);
	allocateMemoryOnGPU(device_M1, device_M2, device_M3);
	copyMatricesToGPU(M1, M2, device_M1, device_M2);
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
	measureExecutionTimes(executionTimes[0], Sequential, device_M1, device_M2, device_M3, gridDim, blockDim);
	measureExecutionTimes(executionTimes[1], Parallel, device_M1, device_M2, device_M3, gridDim, blockDim);
	measureExecutionTimes(executionTimes[2], SharedMemoryAndTiling, device_M1, device_M2, device_M3, gridDim, blockDim);

	// Copy the result matrix from device to host
	cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(float), cudaMemcpyDeviceToHost);

	// Open a new file to write the result into
	char fileName[100];																					  // Max length filename (Just needs to be long enough)
	sprintf(fileName, "Test/Multiplication_Float_Execution_Times_Matrix_Size_%dx%d.csv", M3Rows, M3Cols); // Customize filename to reflect size of result matrix
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
