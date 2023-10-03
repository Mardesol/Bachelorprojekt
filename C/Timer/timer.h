#ifndef TIMER_H
#define TIMER_H

#include <windows.h>
#include <time.h>

typedef struct Timer {
    LARGE_INTEGER begin;
    LARGE_INTEGER end;
    LARGE_INTEGER frequency;
} Timer;


Timer createTimer();
void beginTimer(Timer *timer);
void endTimer(Timer *timer, char *message, size_t messageLength);
double endTimerDouble(Timer *timer);

#endif
