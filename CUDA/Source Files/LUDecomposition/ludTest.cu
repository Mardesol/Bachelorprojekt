#include "ludKernels.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const int numTimesToRun = 20;

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
	void (*function)(float*, int),
	float* device_A, int ADim)
{
	Timer timer = createTimer();
	beginTimer(timer);

	function (device_A, ADim);

	return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureKernelExecutionTimes(
	float *executionTimes,
	void (*kernel)(float *, int),
	float * device_A, int ADim,
	dim3 gridDim, dim3 blockDim)
{
	for (int i = 0; i < numTimesToRun; i++)
	{
		// Measure execution time for the kernel
		float time = measureKernelExecutionTime(kernel, device_A, ADim, gridDim, blockDim);
		executionTimes[i] = time;
	}
}

void measureFunctionExecutionTimes(
	float* executionTimes,
	void (*function)(float*, int),
	float* device_A, int ADim)
{
	for (int i = 0; i < numTimesToRun; i++)
	{
		// Measure execution time for the kernel
		float time = measureFunctionExecutionTime(function, device_A, ADim);
		executionTimes[i] = time;
	}
}

int main(int argc, char* argv[])
{
	int ADim = atoi(argv[1]); 

    size_t memorySize = ADim * ADim * sizeof(float);

	// Timer measure time spent on a process
	Timer timer = createTimer();

    beginTimer(timer);
    Matrix A;
    float *device_A;

    A = createMatrix(ADim, ADim);
    populateWithRandomFloats(A);

    cudaMalloc((void **)&device_A, memorySize);
    cudaMemcpy(device_A, A.data, memorySize, cudaMemcpyHostToDevice);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(16, 16);

	if (ADim <= 16)
	{
		blockDim = dim3(ADim, ADim); // Use matrix size for smaller matrices
	}

	dim3 gridDim((ADim + blockDim.x - 1) / blockDim.x, (ADim + blockDim.y - 1) / blockDim.y);

	// Create an array to store execution times for each kernel
	float executionTimes[3][numTimesToRun]; // 3 kernels, 100 executions each

	// Measure and record execution times
	measureKernelExecutionTimes		(executionTimes[0], Sequential, 			    device_A, ADim, 1, 1);
	measureKernelExecutionTimes		(executionTimes[1], New_Sequential,				device_A, ADim, 1, 1);
	measureFunctionExecutionTimes	(executionTimes[2], Right_Looking_Parallel_LUD, device_A, ADim);

	// Copy the result matrix from device to host
	cudaMemcpy(A.data, device_A, ADim * ADim * sizeof(float), cudaMemcpyDeviceToHost);

	// Open a new file to write the result into
	char fileName[100];																					  // Max length filename (Just needs to be long enough)
	sprintf(fileName, "Test/LUD_Execution_Times_Matrix_Size_%dx%d.csv", ADim, ADim); // Customize filename to reflect size of result matrix
	FILE *outputFile = fopen(fileName, "w");
	if (outputFile == NULL)
	{
		perror("Unable to create the output file");
		return 1;
	}

	// Write execution times to the output file in separate columns
	fprintf(outputFile, "Sequential,SequentialPivot,right_looking_lu\n");
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