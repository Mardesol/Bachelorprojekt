#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\..\Matrix\matrixFloats.cu"
#include "..\..\Timer\timer.cu"

const int M1Rows = 100;
const int M1Cols = 100;
const int M2Rows = 100;
const int M2Cols = 100;
const int M3Rows = M1Rows;
const int M3Cols = M2Cols;

__global__ void MMV1Sequential(float* M1, float* M2, float* M3) {

	for (int i = 0; i < M1Rows; i++) {
		for (int j = 0; j < M2Cols; j++) {
			float sum = 0.0f;
			for (int k = 0; k < M1Cols; k++) {
				sum += M1[i * M1Cols + k] *
					M2[k * M2Cols + j];
			}
			M3[i * M2Cols + j] = sum;
		}
	}
}

__global__ void MMV2Parallelism(float* M1, float* M2, float* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M1Rows && col < M2Cols) {
		float sum = 0.0f;

		for (int i = 0; i < M1Cols; i++) {
			sum += M1[row * M1Cols + i] *
				M2[i * M2Cols + col];
		}
		M3[row * M2Cols + col] = sum;
	}
}

__global__ void MMV3SharedMemoryAndTiling(float* M1, float* M2, float* M3) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate shared memory
	__shared__ float sharedMemory1[16];
	__shared__ float sharedMemory2[16];

	float sum = 0.0f;

	// Read into shared memory in a coalescing manner
	for (int i = 0; i < M1Cols; i += blockDim.x) {
		sharedMemory1[threadIdx.x] = (row < M1Rows && i + threadIdx.x < M1Cols) ? M1[row * M1Cols + i + threadIdx.x] : 0.0f;
		sharedMemory2[threadIdx.y] = (i + threadIdx.y < M1Cols && col < M2Cols) ? M2[(i + threadIdx.y) * M2Cols + col] : 0.0f;

		__syncthreads();

		// Perform the multiplication
		for (int j = 0; j < blockDim.x; j++) {
			sum += sharedMemory1[j] * sharedMemory2[j];
		}

		__syncthreads();
	}

	if (row < M1Rows && col < M2Cols) {
		M3[row * M2Cols + col] = sum;
	}
}

// Function to measure kernel execution time
float measureKernelExecutionTime(
	void (*kernel)(float*, float*, float*),
	float* M1, float* M2, float* M3,
	dim3 gridDim, dim3 blockDim
) {
	Timer timer = createTimer();
	beginTimer(timer);

	cudaDeviceSynchronize();
	kernel << <gridDim, blockDim >> > (M1, M2, M3);
	cudaDeviceSynchronize();

	return endTimerReturnTime(timer);
}

// Function to measure execution times and store them in an array
void measureExecutionTimes(
	float* executionTimes,
	void (*kernel)(float*, float*, float*),
	float* M1, float* M2, float* M3,
	dim3 gridDim, dim3 blockDim
) {
	for (int i = 0; i < 100; i++) {
		// Measure execution time for the kernel
		float time = measureKernelExecutionTime(kernel, M1, M2, M3, gridDim, blockDim);
		executionTimes[i] = time;
	}
}

int main() {
	// Timer measure time spent on a process
	Timer timer = createTimer();

	// Start the setup timer
	beginTimer(timer);

	// Define variables
	MatrixF M1;
	MatrixF M2;
	MatrixF M3;

	// Create the matrix objects
	M1 = createMatrixF(M1Rows, M1Cols);
	M2 = createMatrixF(M2Rows, M2Cols);
	M3 = createMatrixF(M3Rows, M3Cols);

	// Populate the matrices
	populateWithOnesFloats(M1);
	populateWithOnesFloats(M2);

	// Stop the setup timer
	endTimer(timer, "setup");

	// Start the data transfer timer (CPU -> GPU / Host -> Device)
	beginTimer(timer);

	// Create the matrix objects to be stored on the device
	float* device_M1, * device_M2, * device_M3;

	// Allocate memory for matrices on the GPU
	cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(float));
	cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(float));
	cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(float));

	// Copy data from host to device
	cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(float), cudaMemcpyHostToDevice);

	// Stop the data transfer timer (CPU -> GPU / Host -> Device)
	endTimer(timer, "data transfer (CPU -> GPU)");

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(16, 16);

	if (M3Rows <= 16 && M3Cols <= 16) {
		blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
	}

	dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);


	// Create an array to store execution times for each kernel
	float executionTimes[3][100]; // 3 kernels, 100 executions each

	// Measure and record execution times
	measureExecutionTimes(executionTimes[0], MMV1Sequential,				device_M1, device_M2, device_M3, gridDim, blockDim);
	measureExecutionTimes(executionTimes[1], MMV2Parallelism,				device_M1, device_M2, device_M3, gridDim, blockDim);
	measureExecutionTimes(executionTimes[2], MMV3SharedMemoryAndTiling,		device_M1, device_M2, device_M3, gridDim, blockDim);

	// Copy the result matrix from device to host
	cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

	// Deallocate memory on the GPU and CPU
	cudaFree(device_M1);
	cudaFree(device_M2);
	cudaFree(device_M3);

	// Open the output file for writing in append mode
	FILE* outputFile = fopen("Test/Floats-ExecutionTimes.csv", "a");
	if (outputFile == NULL) {
		perror("Unable to open the output file");
		return 1;
	}

	// Write execution times to the output file in separate columns
	fprintf(outputFile, "Sequential,Parallel,SharedMemoryAndTilling\n");
	for (int i = 0; i < 100; i++) {
		fprintf(outputFile, "%f,%f,%f\n",
			executionTimes[0][i],
			executionTimes[1][i],
			executionTimes[2][i]);
	}

	// Close the output file
	fclose(outputFile);

	// Exit program
	return 0;
}

