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

double endTimer(Timer *timer)
{
    QueryPerformanceCounter(&timer->end);
    return (double)(timer->end.QuadPart - timer->begin.QuadPart) / timer->frequency.QuadPart;
}
