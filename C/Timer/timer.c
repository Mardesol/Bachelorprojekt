#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

struct Timer createTimer()
{
	struct Timer timer;
	struct timeval begin, end;
	timer.begin = begin;
	timer.end = end;

	return timer;
}

void beginTimer(struct Timer *timer)
{
	gettimeofday(&timer->begin, 0);
}

void endTimer(struct Timer *timer, char *message, size_t messageLength)
{
	gettimeofday(&timer->end, 0);
	long seconds = timer->end.tv_sec - timer->begin.tv_sec;
	long microseconds = timer->end.tv_usec - timer->begin.tv_usec;
	double elapsed = seconds + microseconds * 1e-6;
	printf("Time spent on %s: %f seconds\n",message, elapsed);
}