#include "timer.h"
#include <stdio.h>

C_Timer create_C_Timer()
{
    C_Timer timer;
    QueryPerformanceFrequency(&timer.frequency);
    return timer;
}

void beginTimer(C_Timer *timer)
{
    QueryPerformanceCounter(&timer->begin);
}

double endTimer(C_Timer *timer)
{
    QueryPerformanceCounter(&timer->end);
    return (double)(timer->end.QuadPart - timer->begin.QuadPart) / timer->frequency.QuadPart;
}
