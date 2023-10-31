#ifndef TIMER_CUH
#define TIMER_CUH

#include "cuda_runtime.h"

typedef struct Timer
{
	cudaEvent_t start;
	cudaEvent_t stop;
} Timer;

Timer createTimer();
void beginTimer(Timer timer);
void endTimer(Timer timer, const char *message, bool printDebug);

#endif