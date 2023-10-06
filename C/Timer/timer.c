#include "timer.h"
#include <stdio.h>

Timer createTimer()
{
    Timer timer;
    QueryPerformanceFrequency(&timer.frequency);
    return timer;
}

void beginTimer(Timer *timer)
{
    QueryPerformanceCounter(&timer->begin);
}

void endTimer(Timer *timer, char *message, size_t messageLength)
{
    QueryPerformanceCounter(&timer->end);
    double elapsed = (double)(timer->end.QuadPart - timer->begin.QuadPart) / timer->frequency.QuadPart;
    printf("Time spent on %s: %f seconds\n", message, elapsed);
}

double endTimerDouble(Timer *timer)
{
    QueryPerformanceCounter(&timer->end);
    return (double)(timer->end.QuadPart - timer->begin.QuadPart) / timer->frequency.QuadPart;
}
