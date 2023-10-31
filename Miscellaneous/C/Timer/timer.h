#ifndef TIMER_H
#define TIMER_H

#include <windows.h>
#include <time.h>

typedef struct C_Timer
{
    LARGE_INTEGER begin;
    LARGE_INTEGER end;
    LARGE_INTEGER frequency;
} C_Timer;

C_Timer create_C_Timer();
void beginTimer(C_Timer *timer);
double endTimer(C_Timer *timer);

#endif
