#include "multiplicationIntsKernels.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;
const int TILE_WIDTH = 16;

inline int min_val(int a, int b) {
	return (a < b) ? a : b;
}

int main(int argc, char* argv[]) {
	if (!isCompatibleForMultiplication(M1Cols, M2Rows)) {
		perror("Matrices must be compatible");
		return 1;
	}

	// Timer measure time spent on a process
	Timer timer = createTimer();

    beginTimer(timer);              
    MatrixI M1, M2, M3;
    int* device_M1, * device_M2, * device_M3;
    initializeMatricesAndMemory(M1, M2, M3);
    allocateMemoryOnGPU(device_M1, device_M2, device_M3);
    copyMatricesToGPU(M1, M2, device_M1, device_M2);
    endTimer(timer, "initialize matrices on CPU and GPU", printDebugMessages);

	// Define block and grid dimensions for CUDA kernel
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

	if (M3Rows <= TILE_WIDTH && M3Cols <= TILE_WIDTH) {
		blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
	}

	dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

	// Convert the command line argument to an integer
	int choice = atoi(argv[1]);

	const char* kernelName;

	if (choice == 1) {
		// Time the matrix multiplication	
		kernelName = "Sequential";                                                                                  // Should reflect the chosen kernel, to name output file accordingly
		beginTimer(timer);
		Sequential << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);									// Launch the CUDA kernel to perform matrix addition
		endTimer(timer, "matrix multiplication (GPU)", printDebugMessages);
	}
	else if (choice == 2) {
		// Time the matrix multiplication
		kernelName = "Parallel";                                                                                   // Should reflect the chosen kernel, to name output file accordingly
		beginTimer(timer);
		Parallel << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);                                     // Launch the CUDA kernel to perform matrix addition
		endTimer(timer, "matrix multiplication (GPU)", printDebugMessages);
	}
	else if (choice == 3) {
		// Time the matrix multiplication
		kernelName = "SharedMemoryAndTiling";                                                                       // Should reflect the chosen kernel, to name output file accordingly
		beginTimer(timer);
		cudaDeviceSynchronize();
		SharedMemoryAndTiling << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);                         // Launch the CUDA kernel to perform matrix addition
		cudaDeviceSynchronize();
		endTimer(timer, "matrix multiplication (GPU)", printDebugMessages);
	}
	
	// Copy the result matrix from device to host
    cudaMemcpy(M3.data, device_M3, memorySize3, cudaMemcpyDeviceToHost);

	// Open a new file to write the result into
	char fileName[100];                                                                                                     // Max length filename (Just needs to be long enough)
	sprintf(fileName, "Test/Multiplication_%s_Ints_Runtime_Matrix_Size_%dx%d.csv", kernelName, M3Rows, M3Cols);				// Customize filename to reflect size of result matrix
	printMatrixToFileInts(fileName, M3);

	//Setup a CPU comparison matrix
	MatrixI MCPU = createMatrixInts(M3Rows, M3Cols);
	multiplicationInts(M1, M2, MCPU);

	//Validate result by comparing to CPU calculations
	bool valid = compareMatricesInts(MCPU, M3);
	if (valid) {
		printf("Matrix multiplication results match!\n");
	}
	else {
		printf("Matrix multiplication results do not match.\n");
		// Write the CPU matrix to text file for analysis
		char fileNameCPU[100];
		sprintf(fileNameCPU, "resultsIntsCPU.txt");

		printMatrixToFileInts(fileNameCPU, MCPU);
	}

	freeMemory(device_M1, device_M2, device_M3, M1, M2, M3);

	// Exit program
	return 0;
}