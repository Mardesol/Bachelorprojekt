#include "timer.cuh"

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

void endTimer(Timer timer, char* message)
{
	float timeElapsed;
	cudaEventRecord(timer.stop, 0);
	cudaEventSynchronize(timer.stop);
	cudaEventElapsedTime(&timeElapsed, timer.start, timer.stop);
	printf("Time spent on %s: %f seconds\n", message, timeElapsed);
}