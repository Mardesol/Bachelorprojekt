#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

typedef struct Timer {
	struct timeval begin;
	struct timeval end;
} Timer;

Timer createTimer();
void beginTimer(Timer *timer);
void endTimer(Timer *timer, char *message, size_t messageLength);
float endTimerFloat(struct Timer *timer);

#endif