#ifndef TIMER_CUH
#define TIMER_CUH

typedef struct Timer {
	cudaEvent_t start;
	cudaEvent_t stop;
} Timer;

Timer createTimer();
void beginTimer(Timer timer);
void endTimer(Timer timer, char *message);

#endif