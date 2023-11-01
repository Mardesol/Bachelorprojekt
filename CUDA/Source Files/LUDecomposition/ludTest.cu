#include "ludKernels.cu"
#include "..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

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

// Function to measure execution times and store them in an array
void measureExecutionTimes(
	float *executionTimes,
	void (*kernel)(float *, int),
	float *A, int n,
	dim3 gridDim, dim3 blockDim)
{
	for (int i = 0; i < 100; i++)
	{
		// Measure execution time for the kernel
		float time = measureKernelExecutionTime(kernel, A, n, gridDim, blockDim);
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
    MatrixF A;
    float *device_A;

    A = createMatrixFloats(ADim, ADim);
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
	float executionTimes[3][100]; // 3 kernels, 100 executions each

	// Measure and record execution times
	measureExecutionTimes(executionTimes[0], LUD_Sequential, 			        device_A, ADim, gridDim, blockDim);
	measureExecutionTimes(executionTimes[1], LUD_Sequential_Partial_Pivoting,   device_A, ADim, gridDim, blockDim);
	measureExecutionTimes(executionTimes[2], LUD_Block,                         device_A, ADim, gridDim, blockDim);

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
	fprintf(outputFile, "Sequential,SequentialPivot,Block\n");
	for (int i = 0; i < 100; i++)
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