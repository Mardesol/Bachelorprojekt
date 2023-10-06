#include "multiplicationDoublesKernels.cu"
#include "..\..\Matrix\matrixOperationsCPU.cu"
#include "..\..\Matrix\matrixCompatability.cu"

const bool printDebugMessages = false;

int main(int argc, char* argv[]) {
    if (!multiplicationCheck(M1Cols, M2Rows)) {
        perror("Matrices must be compatible");
        return 1;
    }

    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Create the matrix objects
    MatrixD M1 = createMatrixDoubles(M1Rows, M1Cols);
    MatrixD M2 = createMatrixDoubles(M2Rows, M2Cols);
    MatrixD M3 = createMatrixDoubles(M3Rows, M3Cols);

    // Populate the matrices
    populateWithRandomDoubles(M1);  
    populateWithRandomDoubles(M2);

    // Stop the setup timer
    endTimer(timer, "setup", printDebugMessages);

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Create the matrix objects to be stored on the device
    double* device_M1, * device_M2, * device_M3;  

    // Allocate memory for matrices on the GPU
    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(double));  
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(double));  
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(double));  

    // Copy data from host to device
    // The data is matrix 1 and 2
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(double), cudaMemcpyHostToDevice);  

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)", printDebugMessages);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
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
        SharedMemoryAndTiling << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);                         // Launch the CUDA kernel to perform matrix addition
        endTimer(timer, "matrix multiplication (GPU)", printDebugMessages);
    }

    // Time transfer from device to host
    beginTimer(timer);                                                                                                      // Start the data transfer timer (GPU -> CPU / Device -> Host)
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(double), cudaMemcpyDeviceToHost);                               // Copy the result matrix from device to host
    endTimer(timer, "data transfer (GPU -> CPU)", printDebugMessages);                                                      // Stop the data transfer timer (GPU -> CPU / Device -> Host)

    // Open a new file to write the result into
    char fileName[100];                                                                                                     // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Multiplication_%s_Doubles_Runtime_Matrix_Size_%dx%d.csv", kernelName, M3Rows, M3Cols);          // Customize filename to reflect size of result matrix
    FILE* outputFile = fopen(fileName, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < M3Rows; i++) {
        for (int j = 0; j < M3Cols; j++) {
            fprintf(outputFile, "%lf ", M3.data[i * M3Rows + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the result file
    fclose(outputFile);

    // Deallocate memory on the GPU and CPU
    cudaFree(device_M1);
    cudaFree(device_M2);
    cudaFree(device_M3);

    // Setup a CPU comparison matrix
    MatrixD MCPU = createMatrixD(M3Rows, M3Cols);
    additionDouble(M1.data, M2.data, MCPU.data, M3Rows, M3Cols);

    //Validate result by comparing to CPU calculations
    bool valid = compareMatricesDouble(MCPU.data, M3.data, M3Rows, M3Cols);
    if (valid) {
        printf("Matrix multiplication results match!\n");
    }
    else {
        printf("Matrix multiplication results do not match.\n");
        // Write the matrices to text files for analysis
        FILE* outputFile1 = fopen("resultDoubleCPU.txt", "w");
        if (outputFile1 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile1, "%lf ", MCPU.data[i * M3Rows + j]);  // Change format specifier to %lf for double
            }
            fprintf(outputFile1, "\n");
        }

        // Close the result file
        fclose(outputFile1);

        FILE* outputFile2 = fopen("resultDoubleGPU.txt", "w");
        if (outputFile2 == NULL) {
            perror("Unable to create the output file");
            return 1;
        }

        // Write host_M3 to the result file
        for (int i = 0; i < M3Rows; i++) {
            for (int j = 0; j < M3Cols; j++) {
                fprintf(outputFile2, "%lf ", M3.data[i * M3Rows + j]);  // Change format specifier to %lf for double
            }
            fprintf(outputFile2, "\n");
        }

        // Close the result file
        fclose(outputFile2);
    }

    // Exit program
    return 0;
}