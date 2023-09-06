#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#include "..\..\Header Files\timer.cuh"

struct Timer createTimer()
{
	struct Timer timer;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	timer.start = start;
	cudaEventCreate(&stop);
	timer.stop = stop;
	return timer;
}

void beginTimer(Timer timer)
{	
	cudaEventRecord(timer.start, 0);
}

void endTimer(Timer timer, const char* message)
{
	float timeElapsed;
	cudaEventRecord(timer.stop, 0);
	cudaEventSynchronize(timer.stop);
	cudaEventElapsedTime(&timeElapsed, timer.start, timer.stop);
	printf("Time spent on %s: %f ms\n", message, timeElapsed);
}