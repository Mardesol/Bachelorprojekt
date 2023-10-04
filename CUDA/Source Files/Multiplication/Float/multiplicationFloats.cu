#include "multiplicationFloatsKernels.cu"

const bool printDebugMessages = false;

int main() {
    // Timer measure time spent on a process
    Timer timer = createTimer();

    // Start the setup timer
    beginTimer(timer);

    // Create the matrix objects
    MatrixF M1 = createMatrixF(M1Rows, M1Cols);
    MatrixF M2 = createMatrixF(M2Rows, M2Cols);
    MatrixF M3 = createMatrixF(M3Rows, M3Cols);

    // Populate the matrices
    populateWithRandomFloats(M1);
    populateWithRandomFloats(M2);

    // Stop the setup timer
    endTimer(timer, "setup", printDebugMessages);

    // Start the data transfer timer (CPU -> GPU / Host -> Device)
    beginTimer(timer);

    // Create the matrix objects to be stored on the device
    float* device_M1, * device_M2, * device_M3;

    // Allocate memory for matrices on the GPU
    cudaMalloc((void**)&device_M1, M1Rows * M1Cols * sizeof(float));
    cudaMalloc((void**)&device_M2, M2Rows * M2Cols * sizeof(float));
    cudaMalloc((void**)&device_M3, M3Rows * M3Cols * sizeof(float));

    // Copy data from host to device
    // The data is matrix 1 and 2
    cudaMemcpy(device_M1, M1.data, M1Rows * M1Cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_M2, M2.data, M2Rows * M2Cols * sizeof(float), cudaMemcpyHostToDevice);

    // Stop the data transfer timer (CPU -> GPU / Host -> Device)
    endTimer(timer, "data transfer (CPU -> GPU)", printDebugMessages);

    // Define block and grid dimensions for CUDA kernel
    dim3 blockDim(16, 16);

    if (M3Rows <= 16 && M3Cols <= 16) {
        blockDim = dim3(M3Cols, M3Rows);  // Use matrix size for smaller matrices
    }

    dim3 gridDim((M3Cols + blockDim.x - 1) / blockDim.x, (M3Rows + blockDim.y - 1) / blockDim.y);

    // Time the matrix multiplication
    const char* kernelName;
    kernelName = "SharedMemoryAndTiling";                                                                                   // Should reflect the chosen kernel, to name output file accordingly
    beginTimer(timer);
    SharedMemoryAndTiling << <gridDim, blockDim >> > (device_M1, device_M2, device_M3);                                     // Launch the CUDA kernel to perform matrix addition
    endTimer(timer, "matrix addition (GPU)", printDebugMessages);

    // Time transfer from device to host
    beginTimer(timer);                                                                                                      // Start the data transfer timer (GPU -> CPU / Device -> Host)
    cudaMemcpy(M3.data, device_M3, M3Rows * M3Cols * sizeof(float), cudaMemcpyDeviceToHost);                                // Copy the result matrix from device to host
    endTimer(timer, "data transfer (GPU -> CPU)", printDebugMessages);                                                      // Stop the data transfer timer (GPU -> CPU / Device -> Host)

    // Open a new file to write the result into
    char fileName[100];                                                                                                     // Max length filename (Just needs to be long enough)
    sprintf(fileName, "Test/Multiplication_%s_Floats_Runtime_Matrix_Size_%dx%d.csv", kernelName, M3Rows, M3Cols);           // Customize filename to reflect size of result matrix
    FILE* outputFile = fopen(fileName, "w");
    if (outputFile == NULL) {
        perror("Unable to create the output file");
        return 1;
    }

    // Write host_M3 to the result file
    for (int i = 0; i < M3Rows; i++) {
        for (int j = 0; j < M3Cols; j++) {
            fprintf(outputFile, "%f ", M3.data[i * M3Rows + j]);
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