#include "ludKernels.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const int numTimesToRun = 100;

// Function to measure kernel execution time
float measureKernelExecutionTime(
	void (*kernel)(float*, int),
	float* device_A, int ADim,
	dim3 gridDim, dim3 blockDim)
{
	Timer timer = createTimer();
	beginTimer(timer);

	cudaDeviceSynchronize();
	kernel<<<gridDim, blockDim>>>(device_A, ADim);
	cudaDeviceSynchronize();

	return endTimerReturnTime(timer);
}

// Function to measure kernel execution time
float measureFunctionExecutionTime(
	int* (*function)(float*, int, dim3),
	float* device_A, int ADim, dim3 blockDim)
{
	Timer timer = createTimer();
	beginTimer(timer);

	function (device_A, ADim, blockDim);

	return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureKernelExecutionTimes(
	float *executionTimes,
	void (*kernel)(float *, int),
	float * device_A, int ADim,
	dim3 gridDim, dim3 blockDim,
	Matrix A, size_t memorySize)
{

	for (int i = 0; i < numTimesToRun; i++)
	{
		// Measure execution time for the kernel
		cudaMemcpy(device_A, A.data, memorySize, cudaMemcpyHostToDevice);
		float time = measureKernelExecutionTime(kernel, device_A, ADim, gridDim, blockDim);
		executionTimes[i] = time;
	}
}

void measureFunctionExecutionTimes(
	float* executionTimes,
	int* (*function)(float*, int, dim3),
	float* device_A, int ADim, dim3 blockDim,
	Matrix A, size_t memorySize)
{

	for (int i = 0; i < numTimesToRun; i++)
	{
		// Measure execution time for the function
		cudaMemcpy(device_A, A.data, memorySize, cudaMemcpyHostToDevice);
		float time = measureFunctionExecutionTime(function, device_A, ADim, blockDim);
		executionTimes[i] = time;
	}
}

int main(int argc, char* argv[])
{
	int ADim = atoi(argv[1]); 
    size_t memorySize = ADim * ADim * sizeof(float);
	Timer timer = createTimer();

    beginTimer(timer);
    Matrix A;
    float *device_A;

    A = createMatrix(ADim, ADim);
    populateWithRandomFloats(A);

    cudaMalloc((void **)&device_A, memorySize);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(32, 32);

	if (ADim <= 32)
	{
		blockDim = dim3(ADim, ADim); // Use matrix size for smaller matrices
	}

	dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

	// Create an array to store execution times for each kernel
	float executionTimes[3][numTimesToRun]; // 3 kernels, 100 executions each

	// Measure and record execution times
	measureKernelExecutionTimes		(executionTimes[0], Sequential_With_Partial_Pivoting,		device_A, ADim, 1, 1, A, memorySize);
	measureFunctionExecutionTimes	(executionTimes[1], Parallel_Pivoted,						device_A, ADim, blockDim, A, memorySize);
	measureFunctionExecutionTimes	(executionTimes[2], SharedMemory_Pivoted,					device_A, ADim, blockDim, A, memorySize);

	// Open a new file to write the result into
	char fileName[100];																 // Max length filename (Just needs to be long enough)
	sprintf(fileName, "Test/LUD_Execution_Times_Matrix_Size_%dx%d.csv", ADim, ADim); // Customize filename to reflect size of result matrix
	FILE *outputFile = fopen(fileName, "w");
	if (outputFile == NULL)
	{
		perror("Unable to create the output file");
		return 1;
	}

	// Write execution times to the output file in separate columns
	fprintf(outputFile, "Sequential,Parallel,Shared Memory\n");
	for (int i = 0; i < numTimesToRun; i++)
	{
		fprintf(outputFile, "%f,%f,%f\n",
				executionTimes[0][i],
				executionTimes[1][i],
				executionTimes[2][i]);
	}

	// Close the output file
	fclose(outputFile);

    free(A.data);
    cudaFree(device_A);

	// Exit program
	return 0;
}