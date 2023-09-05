#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "..\Matrix\matrix.cu"
#include "..\Timer\timer.cu"

__global__ void matrixMultiplicationSimple(int* M1, int* M2, int* M3, int M1R, int M1C, int M2R, int M2C) {
	
	for (int i = 0; i < M1R; i++) {
		for (int j = 0; j < M2C; j++) {
			int sum = 0;
			
			for (int k = 0; k < M1C; k++) {
				int a = M1[i * M1C + k];
				int b = M2[k * M2C + j];
				sum = sum + (a * b);
			}

			M3[i * M2C + j] = sum;
		}
	}
}

__global__ void matrixMultiplicationV2(int* M1, int* M2, int* M3, int M1R, int M1C, int M2R, int M2C) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M1R && col < M2C) {
		int sum = 0;

		for (int k = 0; k < M1C; k++) {
			int a = M1[row * M1C + k];
			int b = M2[k * M2C + col];
			sum += (a * b);
		}
		M3[row * M2C + col] = sum;
	}
}

int main() {

	// Timer measure time spent on a process
	Timer timer = createTimer();

	// Start the setup timer
	beginTimer(timer);

	// Define variables
	Matrix M1;
	Matrix M2;
	Matrix M3;
	int M1Rows = 400;
	int M1Cols = 400;
	int M2Rows = 400;
	int M2Cols = 400;
	int M3Rows = M1Rows;
	int M3Cols = M2Cols;

	// Create the matrix objects
	M1 = createMatrix(M1Rows, M1Cols);
	M2 = createMatrix(M2Rows, M2Cols);
	M3 = createMatrix(M3Rows, M3Cols);

	// Populate the matrices
	populateWithOnes(M1);
	populateWithOnes(M2);

	// Stop the setup timer
	endTimer(timer, "setup");

	// Start the data transfer timer (CPU -> GPU / Host -> Device)
	beginTimer(timer);

	// Create the matrix objects to be stored on the device
	int* device_M1, * device_M2, * device_M3;

	// Allocate memory for matrices on the GPU
	cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(int));
	cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(int));
	cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(int));

	// Copy data from host to device
	// The data is matrix 1 and 2
	cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(int), cudaMemcpyHostToDevice);

	// Stop the data transfer timer (CPU -> GPU / Host -> Device)
	endTimer(timer, "data transfer (CPU -> GPU)");

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(16, 16);
	dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

	// Start the matrix addition timer
	beginTimer(timer);

	// Launch the CUDA kernel to perform matrix multiplication
	matrixMultiplicationSimple <<<gridDim, blockDim>>>(device_M1, device_M2, device_M3, M1Rows, M1Cols, M2Rows, M2Cols);

	// Stop the matrix multiplication timer
	endTimer(timer, "matrix addition (GPU)");

	// Start the data transfer timer (GPU -> CPU / Device -> Host)
	beginTimer(timer);

	// Copy the result matrix from device to host
	cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(int), cudaMemcpyDeviceToHost);

	// Stop the data transfer timer (GPU -> CPU / Device -> Host)
	endTimer(timer, "data transfer (GPU -> CPU)");

	// Open a new file to write the result into
	FILE* outputFile = fopen("result.txt", "w");
	if (outputFile == NULL) {
		perror("Unable to create the output file");
		return 1;
	}

	// Write host_M3 to the result file
	for (int i = 0; i < M3Rows; i++) {
		for (int j = 0; j < M3Cols; j++) {
			fprintf(outputFile, "%d ", M3.data[i * M3Rows + j]);
		}
		fprintf(outputFile, "\n");
	}

	// Close the result file
	fclose(outputFile);

	// Deallocate memory on the GPU and CPU
	cudaFree(device_M1);
	cudaFree(device_M2);
	cudaFree(device_M3);

	// Exit program
	return 0;
}
